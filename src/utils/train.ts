import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

// Define the structure for our training data
interface TrainingExample {
  image: tf.Tensor3D;
  boxes: number[][];
  labels: string[];
  isViolent: boolean;
}

interface TrainingCallbacks {
  onEpochEnd?: (epoch: number, logs: tf.Logs) => void;
  onBatchEnd?: (batch: number, logs: tf.Logs) => void;
  onTrainingEnd?: (logs: tf.Logs) => void;
}

export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
  validationSplit: number;
}

// Default configuration
const DEFAULT_CONFIG: TrainingConfig = {
  learningRate: 0.0001,
  epochs: 50,
  batchSize: 32,
  validationSplit: 0.2
};

const MODEL_KEY = 'violence-detection-model';

// Create a custom model for violence detection
function createCustomModel(): tf.LayersModel {
  const model = tf.sequential();
  
  // Input layer
  model.add(tf.layers.conv2d({
    inputShape: [224, 224, 3],
    kernelSize: 3,
    filters: 32,
    strides: 2,
    activation: 'relu',
    padding: 'same'
  }));
  
  // Feature extraction layers
  model.add(tf.layers.batchNormalization({}));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 64,
    strides: 1,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.batchNormalization({}));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 128,
    strides: 1,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.batchNormalization({}));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
  
  // Classification layers
  model.add(tf.layers.flatten({}));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
  model.add(tf.layers.batchNormalization({}));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  return model;
}

// Prepare training data
async function prepareTrainingData(examples: TrainingExample[]) {
  const xs: tf.Tensor[] = [];
  const ys: tf.Tensor[] = [];

  for (const example of examples) {
    try {
      // Ensure the image tensor is the correct shape
      const normalizedImage = tf.tidy(() => {
        // Ensure the image is 3D [height, width, channels]
        let processedImage = example.image;
        if (processedImage.shape.length === 4) {
          processedImage = tf.squeeze(processedImage, [0]);
        }
        
        // Ensure correct dimensions
        if (processedImage.shape[0] !== 224 || processedImage.shape[1] !== 224) {
          processedImage = tf.image.resizeBilinear(processedImage, [224, 224]);
        }
        
        return processedImage.toFloat().div(255);
      });

      xs.push(normalizedImage);
      ys.push(tf.scalar(example.isViolent ? 1 : 0));
    } catch (error) {
      console.error('Error preparing training example:', error);
      continue;
    }
  }

  if (xs.length === 0) {
    throw new Error('No valid training examples could be prepared');
  }

  // Stack all examples into a single tensor
  const xsStacked = tf.stack(xs);
  const ysStacked = tf.stack(ys);

  // Clean up individual tensors
  xs.forEach(t => t.dispose());
  ys.forEach(t => t.dispose());

  return {
    xs: xsStacked,
    ys: ysStacked
  };
}

// Training function with callbacks
async function trainModel(
  model: tf.LayersModel,
  trainingData: { xs: tf.Tensor; ys: tf.Tensor },
  config: TrainingConfig = DEFAULT_CONFIG,
  callbacks?: TrainingCallbacks
) {
  const optimizer = tf.train.adam(config.learningRate);

  model.compile({
    optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  let lastEpochLogs: tf.Logs | null = null;

  try {
    const history = await model.fit(trainingData.xs, trainingData.ys, {
      batchSize: config.batchSize,
      epochs: config.epochs,
      validationSplit: config.validationSplit,
      shuffle: true,
      callbacks: {
        onEpochBegin: async (epoch) => {
          await tf.nextFrame();
        },
        onEpochEnd: async (epoch, logs) => {
          lastEpochLogs = logs;
          if (callbacks?.onEpochEnd) {
            await tf.nextFrame();
            callbacks.onEpochEnd(epoch, logs);
          }
        },
        onBatchEnd: async (batch, logs) => {
          if (callbacks?.onBatchEnd) {
            await tf.nextFrame();
            callbacks.onBatchEnd(batch, logs);
          }
        },
        onTrainEnd: async (logs) => {
          if (callbacks?.onTrainingEnd) {
            await tf.nextFrame();
            // Ensure final epoch is counted
            callbacks.onEpochEnd?.(config.epochs - 1, lastEpochLogs || logs);
            callbacks.onTrainingEnd(logs);
          }
        }
      }
    });

    return model;
  } catch (error) {
    console.error('Training error:', error);
    throw new Error('Failed to train the model. Please check the training data and try again.');
  }
}

// Save the trained model to IndexedDB
async function saveModel(model: tf.LayersModel) {
  try {
    await model.save(`indexeddb://${MODEL_KEY}`);
  } catch (error) {
    console.error('Error saving model:', error);
    throw new Error('Failed to save the trained model');
  }
}

// Main training pipeline
export async function trainViolenceDetection(
  trainingData: TrainingExample[],
  config: Partial<TrainingConfig> = {},
  callbacks?: TrainingCallbacks
) {
  if (!trainingData || trainingData.length === 0) {
    throw new Error('No training data provided');
  }

  const finalConfig = { ...DEFAULT_CONFIG, ...config };
  let preparedData: { xs: tf.Tensor; ys: tf.Tensor } | null = null;
  let model: tf.LayersModel | null = null;

  try {
    console.log('Creating custom model...');
    model = createCustomModel();

    console.log('Preparing training data...');
    preparedData = await prepareTrainingData(trainingData);

    console.log('Starting training...');
    const trainedModel = await trainModel(model, preparedData, finalConfig, callbacks);

    console.log('Saving model...');
    await saveModel(trainedModel);

    console.log('Training complete!');
    return trainedModel;
  } catch (error) {
    console.error('Error during training:', error);
    throw error;
  } finally {
    // Clean up tensors
    if (preparedData) {
      preparedData.xs.dispose();
      preparedData.ys.dispose();
    }
    // Don't dispose the model as it's being returned
    await tf.nextFrame(); // Ensure UI updates
  }
}

// Function to load the trained model from IndexedDB
export async function loadTrainedModel() {
  try {
    const model = await tf.loadLayersModel(`indexeddb://${MODEL_KEY}`);
    return model;
  } catch (error) {
    console.error('Error loading trained model:', error);
    throw error;
  }
}

// Function to evaluate model performance
export async function evaluateModel(
  model: tf.LayersModel,
  testData: TrainingExample[]
) {
  const { xs, ys } = await prepareTrainingData(testData);
  
  try {
    const evaluation = await model.evaluate(xs, ys);
    
    return {
      loss: Array.isArray(evaluation) ? evaluation[0].dataSync()[0] : evaluation.dataSync()[0],
      accuracy: Array.isArray(evaluation) ? evaluation[1].dataSync()[0] : null
    };
  } finally {
    // Clean up tensors
    xs.dispose();
    ys.dispose();
  }
}