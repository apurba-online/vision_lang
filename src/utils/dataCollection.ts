import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

export interface TrainingExample {
  image: tf.Tensor3D;
  boxes: number[][];
  labels: string[];
  isViolent: boolean;
}

class DataCollector {
  private examples: TrainingExample[] = [];
  private static instance: DataCollector;
  private processingAborted: boolean = false;

  private constructor() {}

  static getInstance(): DataCollector {
    if (!DataCollector.instance) {
      DataCollector.instance = new DataCollector();
    }
    return DataCollector.instance;
  }

  abortProcessing() {
    this.processingAborted = true;
  }

  private async validateFrame(videoElement: HTMLVideoElement): Promise<boolean> {
    return new Promise((resolve) => {
      if (videoElement.readyState >= 3 && 
          videoElement.videoWidth > 0 && 
          videoElement.videoHeight > 0) {
        resolve(true);
        return;
      }

      const checkFrame = () => {
        if (videoElement.readyState >= 3 && 
            videoElement.videoWidth > 0 && 
            videoElement.videoHeight > 0) {
          cleanup();
          resolve(true);
        }
      };

      const handleError = () => {
        cleanup();
        resolve(false);
      };

      const cleanup = () => {
        videoElement.removeEventListener('loadeddata', checkFrame);
        videoElement.removeEventListener('error', handleError);
      };

      videoElement.addEventListener('loadeddata', checkFrame);
      videoElement.addEventListener('error', handleError);

      // Also check immediately
      checkFrame();
    });
  }

  async addExample(
    videoElement: HTMLVideoElement,
    boxes: number[][],
    labels: string[],
    isViolent: boolean
  ): Promise<boolean> {
    try {
      // Validate frame before processing
      const isFrameValid = await this.validateFrame(videoElement);
      if (!isFrameValid) {
        console.warn('Invalid video frame, skipping...');
        return false;
      }

      let imageTensor: tf.Tensor3D | null = null;
      
      try {
        imageTensor = tf.tidy(() => {
          // Capture frame as tensor with error checking
          const pixels = tf.browser.fromPixels(videoElement);
          
          // Validate tensor
          if (!pixels || pixels.shape.length !== 3 || 
              pixels.shape[0] === 0 || pixels.shape[1] === 0) {
            throw new Error('Invalid frame capture');
          }

          // Process the frame
          return pixels
            .resizeBilinear([224, 224])
            .toFloat()
            .div(255) as tf.Tensor3D;
        });

        // Additional validation after processing
        if (!imageTensor || imageTensor.shape.length !== 3 || 
            imageTensor.shape[0] !== 224 || imageTensor.shape[1] !== 224) {
          throw new Error('Frame processing failed');
        }

        this.examples.push({
          image: imageTensor,
          boxes,
          labels,
          isViolent
        });

        return true;
      } catch (error) {
        // Clean up tensor if creation failed
        if (imageTensor) {
          imageTensor.dispose();
        }
        console.warn('Error processing frame:', error);
        return false;
      }
    } catch (error) {
      console.warn('Error adding example:', error);
      return false;
    }
  }

  getExamples(): TrainingExample[] {
    return this.examples;
  }

  clear() {
    // Dispose of all tensors before clearing
    this.examples.forEach(example => {
      if (example.image) {
        try {
          example.image.dispose();
        } catch (error) {
          console.warn('Error disposing tensor:', error);
        }
      }
    });
    this.examples = [];
    this.processingAborted = false;
  }

  async saveToFile() {
    try {
      const data = await Promise.all(
        this.examples.map(async (example) => {
          try {
            const imageData = await tf.browser.toPixels(example.image);
            return {
              boxes: example.boxes,
              labels: example.labels,
              isViolent: example.isViolent,
              imageData
            };
          } catch (error) {
            console.warn('Error converting example to pixels:', error);
            return null;
          }
        })
      );

      const validData = data.filter(d => d !== null);
      if (validData.length === 0) {
        throw new Error('No valid examples to save');
      }

      const blob = new Blob([JSON.stringify(validData)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = 'violence-detection-training-data.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error saving data:', error);
      throw error;
    }
  }
}

export const dataCollector = DataCollector.getInstance();

async function validateVideo(videoElement: HTMLVideoElement): Promise<boolean> {
  return new Promise((resolve) => {
    let timeoutId: number;

    const cleanup = () => {
      videoElement.removeEventListener('loadedmetadata', checkVideo);
      videoElement.removeEventListener('error', handleError);
      if (timeoutId) clearTimeout(timeoutId);
    };

    const handleError = () => {
      cleanup();
      resolve(false);
    };

    const checkVideo = () => {
      if (videoElement.videoWidth > 0 && videoElement.videoHeight > 0 && 
          videoElement.duration > 0 && isFinite(videoElement.duration)) {
        cleanup();
        resolve(true);
      } else {
        handleError();
      }
    };

    videoElement.addEventListener('loadedmetadata', checkVideo);
    videoElement.addEventListener('error', handleError);

    // Set a timeout for validation
    timeoutId = window.setTimeout(() => {
      handleError();
    }, 10000); // 10 second timeout

    // Check immediately in case metadata is already loaded
    checkVideo();
  });
}

export async function processVideoForTraining(
  video: File,
  onProgress?: (progress: number) => void,
  isViolent: boolean = false
): Promise<{ success: boolean; error?: string; framesProcessed: number }> {
  return new Promise((resolve) => {
    const videoElement = document.createElement('video');
    videoElement.muted = true;
    videoElement.playsInline = true;
    videoElement.crossOrigin = 'anonymous';

    let frameCount = 0;
    let errorCount = 0;
    const processInterval = 500; // Process a frame every 500ms
    let lastProcessTime = 0;
    let cocoModel: cocoSsd.ObjectDetection | null = null;
    let isProcessing = true;

    const cleanup = () => {
      isProcessing = false;
      if (videoElement.src) {
        URL.revokeObjectURL(videoElement.src);
      }
      videoElement.remove();
    };

    const processFrame = async () => {
      if (!cocoModel || !videoElement.videoWidth || !isProcessing) return;

      const currentTime = videoElement.currentTime * 1000;
      if (currentTime - lastProcessTime >= processInterval) {
        try {
          // Wait for frame to be ready
          if (videoElement.readyState < 3) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }

          // Get person detections
          const detections = await cocoModel.detect(videoElement);
          const personDetections = detections.filter(d => d.class === 'person');

          if (personDetections.length > 0) {
            const success = await dataCollector.addExample(
              videoElement,
              personDetections.map(d => d.bbox),
              personDetections.map(d => d.class),
              isViolent
            );

            if (success) {
              frameCount++;
            } else {
              errorCount++;
            }
          }

          lastProcessTime = currentTime;

          if (onProgress) {
            const progress = (videoElement.currentTime / videoElement.duration) * 100;
            onProgress(Math.min(100, Math.round(progress)));
          }
        } catch (error) {
          console.warn('Frame processing error:', error);
          errorCount++;
        }
      }

      if (!videoElement.ended && !videoElement.paused && isProcessing) {
        requestAnimationFrame(processFrame);
      } else {
        cleanup();
        resolve({
          success: frameCount > 0,
          framesProcessed: frameCount,
          error: errorCount > 0 ? `${errorCount} frames failed to process` : undefined
        });
      }
    };

    videoElement.addEventListener('loadedmetadata', async () => {
      try {
        if (!await validateVideo(videoElement)) {
          cleanup();
          resolve({
            success: false,
            error: 'Invalid video format or corrupted file',
            framesProcessed: 0
          });
          return;
        }

        cocoModel = await cocoSsd.load({
          base: 'lite_mobilenet_v2'
        });

        await videoElement.play();
        processFrame();
      } catch (error) {
        cleanup();
        resolve({
          success: false,
          error: error instanceof Error ? error.message : 'Failed to initialize video processing',
          framesProcessed: 0
        });
      }
    });

    videoElement.addEventListener('error', () => {
      cleanup();
      resolve({
        success: false,
        error: 'Failed to load video file',
        framesProcessed: 0
      });
    });

    try {
      const videoUrl = URL.createObjectURL(video);
      videoElement.src = videoUrl;
    } catch (error) {
      cleanup();
      resolve({
        success: false,
        error: 'Invalid video file',
        framesProcessed: 0
      });
    }
  });
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

  const processFrame = async () => {
    if (videoElement.paused || videoElement.ended) return;

    const timestamp = videoElement.currentTime * 1000;

    try {
      // Get person detections
      const detections = await cocoModel.detect(videoElement);
      const personDetections = detections.filter(d => d.class === 'person');

      if (personDetections.length > 0) {
        // Prepare frame for violence detection
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

        frameTensor.dispose();
        prediction.dispose();

        onPrediction({
          timestamp,
          isViolent,
          confidence,
          detections: personDetections.map(d => ({
            bbox: d.bbox,
            label: d.class,
            score: d.score
          }))
        });
      }

      // Process next frame
      requestAnimationFrame(processFrame);
    } catch (error) {
      console.warn('Frame processing error:', error);
      // Continue processing despite errors in individual frames
      requestAnimationFrame(processFrame);
    }
  };

  // Start processing frames
  await processFrame();
}