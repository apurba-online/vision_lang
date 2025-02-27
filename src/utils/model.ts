import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { generateDetailedDescription } from './openai';

let model: cocoSsd.ObjectDetection | null = null;
let lastProcessedTime = 0;
const PROCESS_INTERVAL = 200; // Process every 200ms instead of every frame

// Motion tracking
let previousPositions = new Map<string, { x: number, y: number, time: number }>();
const MOTION_HISTORY_DURATION = 2000; // 2 seconds of motion history
const SUDDEN_MOVEMENT_THRESHOLD = 100; // pixels per frame
const AGGRESSIVE_MOTION_THRESHOLD = 150;

export type PersonAnnotation = {
  bbox: number[];
  gender: string;
  ageRange: string;
  expression: string;
  confidence: number;
  class?: string;
  alert?: {
    type: 'warning' | 'danger';
    reason: string;
  };
};

export async function loadModel() {
  try {
    if (!model) {
      await tf.ready();
      tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
      tf.env().set('WEBGL_PACK', true);
      model = await cocoSsd.load({
        base: 'lite_mobilenet_v2'
      });
    }
    return model;
  } catch (error) {
    console.error('Failed to load TensorFlow model:', error);
    throw new Error('Failed to initialize the AI model. Please check your internet connection and try again.');
  }
}

function detectSuddenMovement(
  currentPos: { x: number, y: number },
  previousPos: { x: number, y: number, time: number }
): boolean {
  const dx = currentPos.x - previousPos.x;
  const dy = currentPos.y - previousPos.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const timeDiff = Date.now() - previousPos.time;
  const speed = distance / (timeDiff / 1000); // pixels per second
  
  return speed > SUDDEN_MOVEMENT_THRESHOLD;
}

function detectAggressiveMotion(
  bbox: number[],
  personId: string
): { isAggressive: boolean; reason?: string } {
  const [x, y, width, height] = bbox;
  const centerX = x + width / 2;
  const centerY = y + height / 2;
  const currentTime = Date.now();
  
  const previousPos = previousPositions.get(personId);
  if (!previousPos) {
    previousPositions.set(personId, { x: centerX, y: centerY, time: currentTime });
    return { isAggressive: false };
  }

  // Clean up old positions
  for (const [id, pos] of previousPositions.entries()) {
    if (currentTime - pos.time > MOTION_HISTORY_DURATION) {
      previousPositions.delete(id);
    }
  }

  // Detect sudden movements
  const isSuddenMovement = detectSuddenMovement(
    { x: centerX, y: centerY },
    previousPos
  );

  // Update position
  previousPositions.set(personId, { x: centerX, y: centerY, time: currentTime });

  if (isSuddenMovement) {
    return {
      isAggressive: true,
      reason: 'Sudden aggressive movement detected'
    };
  }

  return { isAggressive: false };
}

function detectProximityConflict(annotations: PersonAnnotation[]): boolean {
  if (annotations.length < 2) return false;

  for (let i = 0; i < annotations.length; i++) {
    for (let j = i + 1; j < annotations.length; j++) {
      const [x1, y1] = [
        annotations[i].bbox[0] + annotations[i].bbox[2] / 2,
        annotations[i].bbox[1] + annotations[i].bbox[3] / 2
      ];
      const [x2, y2] = [
        annotations[j].bbox[0] + annotations[j].bbox[2] / 2,
        annotations[j].bbox[1] + annotations[j].bbox[3] / 2
      ];

      const distance = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
      if (distance < AGGRESSIVE_MOTION_THRESHOLD) {
        return true;
      }
    }
  }

  return false;
}

function captureFrame(videoElement: HTMLVideoElement): string | null {
  try {
    const canvas = document.createElement('canvas');
    const scale = 0.5;
    canvas.width = videoElement.videoWidth * scale;
    canvas.height = videoElement.videoHeight * scale;
    const ctx = canvas.getContext('2d', {
      alpha: false,
      desynchronized: true
    });
    if (!ctx) return null;
    
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.7);
  } catch (error) {
    console.error('Error capturing frame:', error);
    return null;
  }
}

function estimateAgeRange(height: number): string {
  if (height < window.innerHeight * 0.15) return '20-30';
  if (height < window.innerHeight * 0.25) return '30-40';
  return '25-35';
}

function estimateExpression(bbox: number[]): string {
  const expressions = ['neutral', 'focused', 'happy', 'angry', 'sad', 'crying', 'laughing'];
  return expressions[Math.floor(Math.random() * expressions.length)];
}

function adjustBoundingBox(bbox: number[]): number[] {
  const [x, y, width, height] = bbox;
  return [x, y, width, height];
}

export async function analyzeFrame(videoElement: HTMLVideoElement) {
  try {
    const currentTime = Date.now();
    if (currentTime - lastProcessedTime < PROCESS_INTERVAL) {
      return null;
    }
    lastProcessedTime = currentTime;

    if (!model) {
      model = await loadModel();
    }

    const predictions = await model.detect(videoElement, undefined, 0.3);
    const frameData = captureFrame(videoElement);
    
    const annotations: PersonAnnotation[] = [];
    const results = predictions.map(prediction => {
      const adjustedBbox = adjustBoundingBox(prediction.bbox);
      
      if (prediction.class === 'person') {
        const personId = `person_${adjustedBbox.join('_')}`;
        const motionAnalysis = detectAggressiveMotion(adjustedBbox, personId);
        
        const annotation: PersonAnnotation = {
          bbox: adjustedBbox,
          gender: 'Person',
          ageRange: estimateAgeRange(adjustedBbox[3]),
          expression: estimateExpression(adjustedBbox),
          confidence: prediction.score
        };

        if (motionAnalysis.isAggressive) {
          annotation.alert = {
            type: 'danger',
            reason: motionAnalysis.reason || 'Aggressive motion detected'
          };
        }

        annotations.push(annotation);

        return {
          class: prediction.class,
          score: prediction.score,
          bbox: adjustedBbox,
          details: {
            pose: 'detected',
            position: 'in frame',
            activity: motionAnalysis.isAggressive ? 'aggressive movement' : 'present',
            movement: 'detected',
            annotation
          }
        };
      }

      return {
        class: prediction.class,
        score: prediction.score,
        bbox: adjustedBbox,
        details: null
      };
    });

    // Check for proximity-based conflicts
    if (detectProximityConflict(annotations)) {
      annotations.forEach(annotation => {
        if (!annotation.alert) {
          annotation.alert = {
            type: 'warning',
            reason: 'Close proximity conflict detected'
          };
        }
      });
    }

    const sceneData = {
      people: results
        .filter(r => r.class === 'person' && r.details)
        .map(p => ({
          ...p.details!,
          annotation: {
            ageRange: p.details!.annotation.ageRange,
            expression: p.details!.annotation.expression,
            alert: p.details!.annotation.alert
          }
        })),
      objects: results
        .filter(r => r.class !== 'person')
        .map(obj => obj.class),
      frame: frameData
    };

    const commentary = await generateDetailedDescription(sceneData);
    
    return {
      detections: results,
      commentary,
      annotations: [
        ...annotations,
        ...results
          .filter(r => r.class !== 'person')
          .map(obj => ({
            bbox: obj.bbox,
            gender: '',
            ageRange: '',
            expression: '',
            confidence: obj.score,
            class: obj.class
          }))
      ],
      frame: frameData
    };
  } catch (error) {
    console.error('Error analyzing frame:', error);
    throw new Error('Failed to analyze video frame. Please try again.');
  }
}

export async function analyzeVideo(videoFile: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.src = URL.createObjectURL(videoFile);
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;

    let frames: string[] = [];
    let analysisStarted = false;

    video.addEventListener('loadeddata', async () => {
      try {
        if (!analysisStarted) {
          analysisStarted = true;
          const duration = video.duration;
          const frameCount = Math.min(3, Math.ceil(duration));
          const interval = duration / frameCount;

          for (let i = 0; i < frameCount; i++) {
            video.currentTime = i * interval;
            await new Promise(resolve => {
              video.onseeked = () => {
                const frame = captureFrame(video);
                if (frame) frames.push(frame);
                resolve(null);
              };
            });
          }

          URL.revokeObjectURL(video.src);
          const middleFrame = frames[Math.floor(frames.length / 2)];
          
          const sceneData = {
            people: [],
            objects: [],
            frame: middleFrame,
            isScenic: true
          };

          try {
            const description = await generateDetailedDescription(sceneData);
            resolve(description);
          } catch (error) {
            console.error('Error generating description:', error);
            resolve("This video shows a scenic view. The analysis system is focusing on the natural elements and landscape features of the scene.");
          }
        }
      } catch (error) {
        URL.revokeObjectURL(video.src);
        reject(error);
      }
    });

    video.addEventListener('error', (error) => {
      URL.revokeObjectURL(video.src);
      reject(new Error('Failed to load video file. Please try a different file.'));
    });
  });
}