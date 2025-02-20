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
  
  // Estimate face position (approximately 1/6 from the top of body)
  const faceHeight = height * 0.2; // Face is about 20% of body height
  const squareSize = faceHeight;
  
  // Position the box 1/8 down from the top of the body
  // This helps center on the face rather than the hair
  const faceY = y + height * 0.125; // Move down 12.5% of body height
  
  // Center the face box horizontally
  const faceX = x + (width - squareSize) / 2;
  
  return [faceX, faceY, squareSize, squareSize];
}

function adjustBoundingBox(bbox: number[], isPerson: boolean): number[] {
  if (isPerson) {
    return createSquareFaceBox(bbox);
  }
  
  // For objects, make the box slightly tighter
  const [x, y, width, height] = bbox;
  const padding = 0.1; // 10% padding
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
  
  return {
    pose: detectPose(bbox),
    position: detectPosition(bbox),
    activity: detectActivity(bbox),
    movement: detectMovement('person', bbox),
    annotation: {
      bbox: adjustedBbox,
      gender: 'Person',
      ageRange: estimateAgeRange(adjustedBbox[3]),
      expression: estimateExpression(adjustedBbox),
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
    
    const results = predictions.map(prediction => {
      const adjustedBbox = prediction.class === 'person' 
        ? prediction.bbox // Person bbox will be adjusted in analyzePersonDetails
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
        .map(p => p.details!)
        .filter(Boolean),
      objects: results
        .filter(r => r.class !== 'person')
        .map(obj => obj.class)
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
      ]
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

    let detections: Array<{ object: string, timestamp: number }> = [];
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
                detections.push({
                  object: result.class,
                  timestamp: currentTime
                });
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

async function summarizeVideoAnalysis(detections: Array<{ object: string, timestamp: number }>, duration: number): Promise<string> {
  if (detections.length === 0) {
    return "I didn't detect any objects in the video. Try adjusting the lighting or camera angle for better results.";
  }

  const objectSummary = detections.reduce((acc, detection) => {
    if (!acc[detection.object]) {
      acc[detection.object] = {
        count: 1,
        firstSeen: detection.timestamp,
        lastSeen: detection.timestamp
      };
    } else {
      acc[detection.object].count++;
      acc[detection.object].lastSeen = detection.timestamp;
    }
    return acc;
  }, {} as Record<string, { count: number; firstSeen: number; lastSeen: number }>);

  const sceneData = {
    people: Object.entries(objectSummary)
      .filter(([object]) => object === 'person')
      .map(([_, data]) => ({
        pose: 'detected',
        position: `from ${data.firstSeen.toFixed(1)}s to ${data.lastSeen.toFixed(1)}s`,
        activity: `appeared ${data.count} times`,
        movement: 'throughout the video'
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
        summary.push(`Person detected from ${data.firstSeen.toFixed(1)}s to ${data.lastSeen.toFixed(1)}s`);
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