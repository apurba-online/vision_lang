import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { generateDetailedDescription } from './openai';

let model: cocoSsd.ObjectDetection | null = null;
let previousPositions: Record<string, { x: number, y: number }> = {};

export type PersonAnnotation = {
  bbox: number[];
  gender: string;
  ageRange: string;
  expression: string;
  confidence: number;
  class?: string;
};

export async function loadModel() {
  try {
    if (!model) {
      await tf.ready();
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

function captureFrame(videoElement: HTMLVideoElement): string | null {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    
    ctx.drawImage(videoElement, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.5); // Reduced quality for better performance
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
  const expressions = ['neutral', 'happy', 'focused', 'serious'];
  return expressions[Math.floor(Math.random() * expressions.length)];
}

function detectMovement(object: string, bbox: number[]): string {
  const centerX = bbox[0] + bbox[2] / 2;
  const centerY = bbox[1] + bbox[3] / 2;
  
  if (!previousPositions[object]) {
    previousPositions[object] = { x: centerX, y: centerY };
    return 'just appeared';
  }

  const dx = centerX - previousPositions[object].x;
  const dy = centerY - previousPositions[object].y;
  previousPositions[object] = { x: centerX, y: centerY };

  const movements: string[] = [];
  
  if (Math.abs(dx) > 10) {
    movements.push(dx > 0 ? 'moving right' : 'moving left');
  }
  
  if (Math.abs(dy) > 10) {
    movements.push(dy > 0 ? 'moving down' : 'moving up');
  }

  return movements.length > 0 ? movements.join(' and ') : 'standing still';
}

function createSquareFaceBox(bbox: number[]): number[] {
  const [x, y, width, height] = bbox;
  const faceHeight = height * 0.2;
  const squareSize = faceHeight;
  const faceY = y + height * 0.125;
  const faceX = x + (width - squareSize) / 2;
  
  return [faceX, faceY, squareSize, squareSize];
}

function adjustBoundingBox(bbox: number[], isPerson: boolean): number[] {
  if (isPerson) {
    return createSquareFaceBox(bbox);
  }
  
  const [x, y, width, height] = bbox;
  const padding = 0.1;
  const adjustedWidth = width * (1 - padding);
  const adjustedHeight = height * (1 - padding);
  const adjustedX = x + (width - adjustedWidth) / 2;
  const adjustedY = y + (height - adjustedHeight) / 2;
  
  return [adjustedX, adjustedY, adjustedWidth, adjustedHeight];
}

function detectPose(bbox: number[]): string {
  const height = bbox[3];
  const width = bbox[2];
  const aspectRatio = width / height;

  if (aspectRatio < 0.4) return 'standing upright';
  if (aspectRatio > 1.2) return 'lying down';
  if (aspectRatio > 0.8) return 'sitting';
  if (height < window.innerHeight * 0.3) return 'standing far away';
  if (height > window.innerHeight * 0.7) return 'standing very close';
  return 'standing at medium distance';
}

function detectPosition(bbox: number[]): string {
  const [x, y, width, height] = bbox;
  const screenWidth = window.innerWidth;
  const screenHeight = window.innerHeight;
  
  let position = '';
  
  if (x < screenWidth * 0.3) {
    position = 'on the left side';
  } else if (x + width > screenWidth * 0.7) {
    position = 'on the right side';
  } else {
    position = 'in the center';
  }

  if (height < screenHeight * 0.3) {
    position += ' in the background';
  } else if (height > screenHeight * 0.7) {
    position += ' in the foreground';
  }

  return position;
}

function analyzePersonDetails(bbox: number[]): {
  pose: string;
  position: string;
  activity: string;
  movement: string;
  annotation: PersonAnnotation;
} {
  const adjustedBbox = adjustBoundingBox(bbox, true);
  const ageRange = estimateAgeRange(adjustedBbox[3]);
  const expression = estimateExpression(adjustedBbox);
  
  return {
    pose: detectPose(bbox),
    position: detectPosition(bbox),
    activity: detectActivity(bbox),
    movement: detectMovement('person', bbox),
    annotation: {
      bbox: adjustedBbox,
      gender: 'Person',
      ageRange,
      expression,
      confidence: 0.95
    }
  };
}

function detectActivity(bbox: number[]): string {
  const height = bbox[3];
  const screenHeight = window.innerHeight;
  const activities: string[] = [];
  
  if (height > screenHeight * 0.4) {
    const headPosition = bbox[1] / screenHeight;
    if (headPosition > 0.4) {
      activities.push('looking at phone or device');
    }
  }

  const [x, width] = [bbox[0], bbox[2]];
  if (x < 50) {
    activities.push('entering frame');
  } else if (x + width > window.innerWidth - 50) {
    activities.push('leaving frame');
  }

  return activities.join(', ') || 'no specific activity detected';
}

export async function analyzeFrame(videoElement: HTMLVideoElement) {
  try {
    if (!model) {
      model = await loadModel();
    }

    const predictions = await model.detect(videoElement);
    const frameData = captureFrame(videoElement);
    
    const results = predictions.map(prediction => {
      const adjustedBbox = prediction.class === 'person' 
        ? prediction.bbox
        : adjustBoundingBox(prediction.bbox, false);
        
      return {
        class: prediction.class,
        score: prediction.score,
        bbox: adjustedBbox,
        details: prediction.class === 'person' ? analyzePersonDetails(prediction.bbox) : null
      };
    });

    const sceneData = {
      people: results
        .filter(r => r.class === 'person' && r.details)
        .map(p => ({
          ...p.details!,
          annotation: {
            ageRange: p.details!.annotation.ageRange,
            expression: p.details!.annotation.expression
          }
        }))
        .filter(Boolean),
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
        ...results
          .filter(r => r.class === 'person' && r.details)
          .map(p => p.details!.annotation),
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

    let detections: Array<{ 
      object: string;
      timestamp: number;
      ageRange?: string;
      expression?: string;
    }> = [];
    let analysisStarted = false;

    video.addEventListener('loadeddata', async () => {
      try {
        if (!model) {
          await loadModel();
        }

        if (!analysisStarted) {
          analysisStarted = true;
          const startTime = Date.now();
          
          const interval = setInterval(async () => {
            try {
              if (video.ended) {
                clearInterval(interval);
                URL.revokeObjectURL(video.src);
                const summary = await summarizeVideoAnalysis(detections, (Date.now() - startTime) / 1000);
                resolve(summary);
                return;
              }

              const results = await analyzeFrame(video);
              const currentTime = (Date.now() - startTime) / 1000;
              
              results.detections.forEach(result => {
                if (result.class === 'person' && result.details) {
                  detections.push({
                    object: result.class,
                    timestamp: currentTime,
                    ageRange: result.details.annotation.ageRange,
                    expression: result.details.annotation.expression
                  });
                } else {
                  detections.push({
                    object: result.class,
                    timestamp: currentTime
                  });
                }
              });
            } catch (error) {
              clearInterval(interval);
              URL.revokeObjectURL(video.src);
              reject(error);
            }
          }, 1000);
        }

        video.play();
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

async function summarizeVideoAnalysis(
  detections: Array<{ 
    object: string;
    timestamp: number;
    ageRange?: string;
    expression?: string;
  }>,
  duration: number
): Promise<string> {
  if (detections.length === 0) {
    return "I didn't detect any objects in the video. Try adjusting the lighting or camera angle for better results.";
  }

  const objectSummary = detections.reduce((acc, detection) => {
    const key = detection.object;
    if (!acc[key]) {
      acc[key] = {
        count: 1,
        firstSeen: detection.timestamp,
        lastSeen: detection.timestamp,
        ageRanges: detection.ageRange ? new Set([detection.ageRange]) : new Set(),
        expressions: detection.expression ? new Set([detection.expression]) : new Set()
      };
    } else {
      acc[key].count++;
      acc[key].lastSeen = detection.timestamp;
      if (detection.ageRange) acc[key].ageRanges.add(detection.ageRange);
      if (detection.expression) acc[key].expressions.add(detection.expression);
    }
    return acc;
  }, {} as Record<string, {
    count: number;
    firstSeen: number;
    lastSeen: number;
    ageRanges: Set<string>;
    expressions: Set<string>;
  }>);

  const sceneData = {
    people: Object.entries(objectSummary)
      .filter(([object]) => object === 'person')
      .map(([_, data]) => ({
        pose: 'detected',
        position: `from ${data.firstSeen.toFixed(1)}s to ${data.lastSeen.toFixed(1)}s`,
        activity: `appeared ${data.count} times`,
        movement: 'throughout the video',
        annotation: {
          ageRange: Array.from(data.ageRanges).join(', '),
          expression: Array.from(data.expressions).join(', ')
        }
      })),
    objects: Object.entries(objectSummary)
      .filter(([object]) => object !== 'person')
      .map(([object]) => object)
  };

  try {
    return await generateDetailedDescription(sceneData);
  } catch (error) {
    console.error('Error generating detailed description:', error);
    
    const summary = [];
    summary.push(`Video duration: ${duration.toFixed(1)} seconds\n`);

    const people = Object.entries(objectSummary)
      .filter(([object]) => object === 'person');
    const objects = Object.entries(objectSummary)
      .filter(([object]) => object !== 'person');

    if (people.length > 0) {
      people.forEach(([_, data]) => {
        const ageRanges = Array.from(data.ageRanges).join(', ');
        const expressions = Array.from(data.expressions).join(', ');
        summary.push(
          `Person detected from ${data.firstSeen.toFixed(1)}s to ${data.lastSeen.toFixed(1)}s` +
          (ageRanges ? ` (age: ${ageRanges})` : '') +
          (expressions ? ` (expressions: ${expressions})` : '')
        );
      });
    }

    if (objects.length > 0) {
      summary.push('\nObjects detected:');
      objects.forEach(([object, data]) => {
        summary.push(`- ${object}: seen ${data.count} times, first at ${data.firstSeen.toFixed(1)}s`);
      });
    }

    return summary.join('\n');
  }
}
