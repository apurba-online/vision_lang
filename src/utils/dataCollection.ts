import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

export interface TrainingExample {
  image: tf.Tensor3D;
  boxes: number[][];
  labels: string[];
  isViolent: boolean;
}

// Add the missing dataCollector singleton
export const dataCollector = {
  examples: [] as TrainingExample[],
  
  addExample(example: TrainingExample) {
    this.examples.push(example);
  },

  getExamples(): TrainingExample[] {
    return this.examples;
  },

  clearExamples() {
    // Cleanup tensors to prevent memory leaks
    this.examples.forEach(example => {
      if (example.image instanceof tf.Tensor) {
        example.image.dispose();
      }
    });
    this.examples = [];
  }
};

export async function processVideoForTraining(
  video: File,
  onProgress: (progress: number) => void,
  isViolent: boolean
): Promise<{ success: boolean; error?: string; framesProcessed: number }> {
  try {
    const videoElement = document.createElement('video');
    videoElement.src = URL.createObjectURL(video);
    await videoElement.play();

    const cocoModel = await cocoSsd.load({
      base: 'lite_mobilenet_v2'
    });

    const frameCount = Math.floor(videoElement.duration * 5); // Process 5 frames per second
    let processedFrames = 0;

    for (let i = 0; i < frameCount; i++) {
      videoElement.currentTime = i / 5;
      
      // Wait for the frame to be loaded
      await new Promise(resolve => {
        videoElement.addEventListener('seeked', resolve, { once: true });
      });

      // Get person detections
      const detections = await cocoModel.detect(videoElement);
      const personDetections = detections.filter(d => d.class === 'person');

      // Create training example
      const frameTensor = tf.tidy(() => {
        return tf.browser.fromPixels(videoElement)
          .resizeBilinear([224, 224])
          .toFloat()
          .div(255);
      });

      dataCollector.addExample({
        image: frameTensor,
        boxes: personDetections.map(d => d.bbox),
        labels: personDetections.map(d => d.class),
        isViolent
      });

      processedFrames++;
      onProgress((processedFrames / frameCount) * 100);
    }

    // Cleanup
    URL.revokeObjectURL(videoElement.src);
    videoElement.remove();

    return {
      success: true,
      framesProcessed: processedFrames
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to process video',
      framesProcessed: 0
    };
  }
}

export async function processVideoForPrediction(
  videoElement: HTMLVideoElement,
  model: tf.LayersModel,
  onPrediction: (prediction: {
    timestamp: number;
    isViolent: boolean;
    confidence: number;
    detections: Array<{
      bbox: number[];
      label: string;
      score: number;
    }>;
  }) => void
): Promise<void> {
  const cocoModel = await cocoSsd.load({
    base: 'lite_mobilenet_v2'
  });

  let isProcessing = true;
  let lastProcessTime = 0;
  const PROCESS_INTERVAL = 100; // Process every 100ms

  const processFrame = async () => {
    if (!isProcessing || videoElement.paused || videoElement.ended) return;

    const now = performance.now();
    if (now - lastProcessTime < PROCESS_INTERVAL) {
      requestAnimationFrame(processFrame);
      return;
    }

    try {
      // Get person detections
      const detections = await cocoModel.detect(videoElement);
      const personDetections = detections.filter(d => d.class === 'person');

      // Process frame for violence detection
      const frameTensor = tf.tidy(() => {
        return tf.browser.fromPixels(videoElement)
          .resizeBilinear([224, 224])
          .toFloat()
          .div(255)
          .expandDims(0);
      });

      // Get violence prediction
      const prediction = await model.predict(frameTensor) as tf.Tensor;
      const confidence = prediction.dataSync()[0];
      const isViolent = confidence > 0.5;

      // Clean up tensors
      frameTensor.dispose();
      prediction.dispose();

      // Send prediction results
      onPrediction({
        timestamp: now,
        isViolent,
        confidence,
        detections: personDetections.map(d => ({
          bbox: d.bbox,
          label: d.class,
          score: d.score
        }))
      });

      lastProcessTime = now;
    } catch (error) {
      console.warn('Frame processing error:', error);
    }

    requestAnimationFrame(processFrame);
  };

  // Start processing frames
  processFrame();

  // Return cleanup function
  return () => {
    isProcessing = false;
  };
}