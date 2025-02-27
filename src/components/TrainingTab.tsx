import React, { useState, useRef } from 'react';
import { Upload, Play, RefreshCw, Save, X, Video, AlertCircle, Loader2, Settings, Check } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { processVideoForTraining, dataCollector } from '../utils/dataCollection';
import { trainViolenceDetection, type TrainingConfig } from '../utils/train';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
}

interface VideoStatus {
  isProcessing: boolean;
  progress: number;
  error?: string;
  isViolent: boolean;
}

interface ProcessingResult {
  success: boolean;
  error?: string;
  framesProcessed: number;
}

const SUPPORTED_VIDEO_FORMATS = {
  'video/mp4': ['.mp4'],
  'video/webm': ['.webm'],
  'video/x-matroska': ['.mkv'],
  'video/quicktime': ['.mov'],
  'video/x-msvideo': ['.avi']
};

const formatBytes = (bytes: number, decimals = 2) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

// Function to determine if a video is violent based on filename
const isViolentVideo = (filename: string): boolean => {
  const lowerName = filename.toLowerCase();
  return lowerName.startsWith('v_') || !lowerName.startsWith('nv_');
};

export function TrainingTab() {
  const [trainingVideos, setTrainingVideos] = useState<File[]>([]);
  const [videoStatus, setVideoStatus] = useState<Map<string, VideoStatus>>(new Map());
  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [processingStats, setProcessingStats] = useState<{
    totalVideos: number;
    processedVideos: number;
    totalFrames: number;
    erroredVideos: number;
  }>({
    totalVideos: 0,
    processedVideos: 0,
    totalFrames: 0,
    erroredVideos: 0
  });
  
  // Training configuration state
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    learningRate: 0.0001,
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2
  });

  const stopProcessing = () => {
    // Stop all video processing
    setVideoStatus(prev => {
      const newStatus = new Map(prev);
      for (const [name, status] of newStatus.entries()) {
        if (status.isProcessing) {
          newStatus.set(name, {
            ...status,
            isProcessing: false,
            error: 'Processing stopped'
          });
        }
      }
      return newStatus;
    });
  };

  const clearUploads = () => {
    // Clear all videos and their status
    setTrainingVideos([]);
    setVideoStatus(new Map());
    setError(null);
  };

  const validateVideoFile = (file: File): boolean => {
    if (file.size > 100 * 1024 * 1024) {
      setError(`File ${file.name} is too large. Maximum size is 100MB`);
      return false;
    }

    const isSupported = Object.keys(SUPPORTED_VIDEO_FORMATS).some(type => {
      if (file.type === type) return true;
      const extensions = SUPPORTED_VIDEO_FORMATS[type as keyof typeof SUPPORTED_VIDEO_FORMATS];
      return extensions.some(ext => file.name.toLowerCase().endsWith(ext));
    });

    if (!isSupported) {
      setError(`Unsupported format for ${file.name}. Please use MP4, WebM, MOV, MKV, or AVI files.`);
      return false;
    }

    return true;
  };

  const onDrop = async (acceptedFiles: File[]) => {
    setError(null);
    const validVideos = acceptedFiles.filter(validateVideoFile);
    if (validVideos.length === 0) return;

    // Initialize status for new videos with automatic classification
    const newStatus = new Map(videoStatus);
    validVideos.forEach(video => {
      newStatus.set(video.name, {
        isProcessing: false,
        progress: 0,
        isViolent: isViolentVideo(video.name) // Auto-classify based on filename
      });
    });
    setVideoStatus(newStatus);
    setTrainingVideos(prev => [...prev, ...validVideos]);

    // Show message about auto-classification
    const violentCount = Array.from(newStatus.values()).filter(s => s.isViolent).length;
    const nonViolentCount = newStatus.size - violentCount;
    setError(`Auto-classified ${violentCount} violent and ${nonViolentCount} non-violent videos based on filenames. You can manually adjust the classifications if needed.`);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: SUPPORTED_VIDEO_FORMATS,
    maxSize: 100 * 1024 * 1024,
    multiple: true
  });

  const toggleVideoClass = (videoName: string) => {
    setVideoStatus(prev => {
      const newStatus = new Map(prev);
      const status = newStatus.get(videoName);
      if (status) {
        newStatus.set(videoName, {
          ...status,
          isViolent: !status.isViolent
        });
      }
      return newStatus;
    });
  };

  const startTraining = async () => {
    if (trainingVideos.length === 0) {
      setError('Please upload at least one video file.');
      return;
    }

    // Reset stats
    setProcessingStats({
      totalVideos: trainingVideos.length,
      processedVideos: 0,
      totalFrames: 0,
      erroredVideos: 0
    });

    setError(null);
    setMetrics([]);
    setCurrentEpoch(0);
    setTrainingComplete(false);

    try {
      const results: ProcessingResult[] = [];

      // Process all videos sequentially
      for (const video of trainingVideos) {
        const status = videoStatus.get(video.name);
        if (!status) continue;

        // Update status to processing
        setVideoStatus(prev => new Map(prev).set(video.name, {
          ...status,
          isProcessing: true,
          progress: 0
        }));

        try {
          const result = await processVideoForTraining(
            video,
            (progress) => {
              setVideoStatus(prev => new Map(prev).set(video.name, {
                ...status,
                isProcessing: true,
                progress
              }));
            },
            status.isViolent
          );

          results.push(result);

          // Update status based on result
          setVideoStatus(prev => new Map(prev).set(video.name, {
            ...status,
            isProcessing: false,
            progress: result.success ? 100 : 0,
            error: result.error
          }));

          // Update processing stats
          setProcessingStats(prev => ({
            ...prev,
            processedVideos: prev.processedVideos + 1,
            totalFrames: prev.totalFrames + result.framesProcessed,
            erroredVideos: prev.erroredVideos + (result.success ? 0 : 1)
          }));

        } catch (err) {
          results.push({
            success: false,
            error: err instanceof Error ? err.message : 'Processing failed',
            framesProcessed: 0
          });

          setVideoStatus(prev => new Map(prev).set(video.name, {
            ...status,
            isProcessing: false,
            progress: 0,
            error: err instanceof Error ? err.message : 'Processing failed'
          }));

          setProcessingStats(prev => ({
            ...prev,
            processedVideos: prev.processedVideos + 1,
            erroredVideos: prev.erroredVideos + 1
          }));
        }
      }

      // Check if we have enough successful videos
      const successfulResults = results.filter(r => r.success);
      const totalFrames = results.reduce((sum, r) => sum + r.framesProcessed, 0);

      if (successfulResults.length === 0) {
        throw new Error('No videos were successfully processed. Please check the format and try again.');
      }

      // Show processing summary
      setError(`Processed ${successfulResults.length} videos successfully with ${totalFrames} frames. ${
        results.length - successfulResults.length
      } videos were skipped due to errors.`);

      // Get training examples from the collector
      const trainingExamples = dataCollector.getExamples();
      if (trainingExamples.length === 0) {
        throw new Error('No valid training examples were generated from the videos.');
      }

      console.log(`Starting training with ${trainingExamples.length} examples`);
      setIsTraining(true);

      // Start actual training with custom configuration
      await trainViolenceDetection(
        trainingExamples,
        trainingConfig,
        {
          onEpochEnd: (epoch, logs) => {
            setCurrentEpoch(epoch + 1);
            setMetrics(prev => [...prev, {
              epoch,
              loss: logs.loss,
              accuracy: logs.acc || 0,
              valLoss: logs.val_loss || logs.loss,
              valAccuracy: logs.val_acc || 0
            }]);
          },
          onTrainingEnd: () => {
            setTrainingComplete(true);
            setCurrentEpoch(trainingConfig.epochs);
            setIsTraining(false);
          }
        }
      );

      console.log('Training completed successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during processing');
      setIsTraining(false);
    }
  };

  const stopTraining = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsTraining(false);
  };

  const removeVideo = (index: number) => {
    const video = trainingVideos[index];
    setTrainingVideos(prev => prev.filter((_, i) => i !== index));
    setVideoStatus(prev => {
      const newStatus = new Map(prev);
      newStatus.delete(video.name);
      return newStatus;
    });
  };

  const chartData = {
    labels: metrics.map(m => `Epoch ${m.epoch + 1}`),
    datasets: [
      {
        label: 'Training Loss',
        data: metrics.map(m => m.loss),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1
      },
      {
        label: 'Validation Loss',
        data: metrics.map(m => m.valLoss),
        borderColor: 'rgb(53, 162, 235)',
        tension: 0.1
      },
      {
        label: 'Training Accuracy',
        data: metrics.map(m => m.accuracy),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      {
        label: 'Validation Accuracy',
        data: metrics.map(m => m.valAccuracy),
        borderColor: 'rgb(153, 102, 255)',
        tension: 0.1
      }
    ]
  };

  return (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Video Upload Section */}
        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Video className="w-5 h-5 text-blue-500" />
              Training Videos
            </h3>
            <div className="flex gap-2">
              {Array.from(videoStatus.values()).some(s => s.isProcessing) && (
                <button
                  onClick={stopProcessing}
                  className="px-3 py-1.5 text-sm bg-red-600 hover:bg-red-700 text-white rounded-lg transition flex items-center gap-1.5"
                >
                  <X size={14} />
                  Stop Processing
                </button>
              )}
              {trainingVideos.length > 0 && (
                <button
                  onClick={clearUploads}
                  className="px-3 py-1.5 text-sm bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition flex items-center gap-1.5"
                >
                  <X size={14} />
                  Clear All
                </button>
              )}
            </div>
          </div>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${
              isDragActive 
                ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-500/10' 
                : 'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto mb-4 text-gray-400 dark:text-gray-500" size={48} />
            <p className="text-sm mb-2">
              {isDragActive
                ? 'Drop the videos here'
                : 'Drag & drop video files here, or click to select'}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Supports MP4, WebM, MOV, MKV, and AVI formats (max 100MB per file)
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Prefix filenames with "NV_" for non-violent or "V_" for violent videos
            </p>
          </div>

          {trainingVideos.length > 0 && (
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium">Uploaded Videos:</h4>
                <span className="text-xs text-gray-500">
                  {trainingVideos.length} video{trainingVideos.length !== 1 ? 's' : ''}
                </span>
              </div>
              <div className="max-h-[200px] overflow-y-auto scrollbar-thin pr-2">
                <div className="space-y-1">
                  {trainingVideos.map((video, index) => {
                    const status = videoStatus.get(video.name);
                    if (!status) return null;
                    
                    return (
                      <div
                        key={index}
                        className={`p-2 rounded-lg ${
                          status.error
                            ? 'bg-red-50 dark:bg-red-900/20'
                            : status.isProcessing
                            ? 'bg-blue-50 dark:bg-blue-900/20'
                            : status.progress === 100
                            ? 'bg-green-50 dark:bg-green-900/20'
                            : 'bg-gray-50 dark:bg-gray-800'
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0 flex-1">
                            <Video className={`w-4 h-4 flex-shrink-0 ${
                              status.error
                                ? 'text-red-500'
                                : status.isProcessing
                                ? 'text-blue-500'
                                : status.progress === 100
                                ? 'text-green-500'
                                : 'text-gray-500'
                            }`} />
                            <div className="min-w-0 flex-1">
                              <div className="flex items-baseline gap-2">
                                <span className="text-sm truncate">{video.name}</span>
                                <span className="text-xs text-gray-500 flex-shrink-0">
                                  ({formatBytes(video.size)})
                                </span>
                              </div>
                              <div className="flex items-center gap-2 mt-1">
                                <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                                  <div
                                    className={`h-1 rounded-full transition-all ${
                                      status.error
                                        ? 'bg-red-500'
                                        : status.progress === 100
                                        ? 'bg-green-500'
                                        : 'bg-blue-500'
                                    }`}
                                    style={{ width: `${status.progress || 0}%` }}
                                  />
                                </div>
                                <span className={`text-xs flex items-center gap-1 ${
                                  status.error
                                    ? 'text-red-500'
                                    : status.isProcessing
                                    ? 'text-blue-500'
                                    : status.progress === 100
                                    ? 'text-green-500'
                                    : 'text-gray-500'
                                }`}>
                                  {status.isProcessing && (
                                    <Loader2 size={10} className="animate-spin" />
                                  )}
                                  {status.error
                                    ? 'Error'
                                    : status.isProcessing
                                    ? `${Math.round(status.progress)}%`
                                    : status.progress === 100
                                    ? 'Done'
                                    : 'Ready'}
                                </span>
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => toggleVideoClass(video.name)}
                              className={`px-2 py-1 text-xs rounded ${
                                status.isViolent
                                  ? 'bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400'
                                  : 'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400'
                              }`}
                              title={status.isViolent ? "Mark as Non-Violent" : "Mark as Violent"}
                            >
                              <Check size={14} className={status.isViolent ? 'text-red-500' : 'text-green-500'} />
                              {status.isViolent ? 'Violent' : 'Non-Violent'}
                            </button>
                            <button
                              onClick={() => removeVideo(index)}
                              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full flex-shrink-0"
                            >
                              <X size={14} />
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg text-sm flex items-center gap-2">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Training Controls */}
        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Training Controls</h3>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition"
              title="Training Settings"
            >
              <Settings size={20} />
            </button>
          </div>

          {showSettings && (
            <div className="mb-6 space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Learning Rate
                </label>
                <input
                  type="number"
                  value={trainingConfig.learningRate}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    learningRate: parseFloat(e.target.value)
                  }))}
                  step="0.0001"
                  min="0.0001"
                  max="0.1"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Epochs
                </label>
                <input
                  type="number"
                  value={trainingConfig.epochs}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    epochs: parseInt(e.target.value)
                  }))}
                  min="1"
                  max="200"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={trainingConfig.batchSize}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    batchSize: parseInt(e.target.value)
                  }))}
                  min="1"
                  max="128"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Validation Split
                </label>
                <input
                  type="number"
                  value={trainingConfig.validationSplit}
                  onChange={(e) => setTrainingConfig(prev => ({
                    ...prev,
                    validationSplit: parseFloat(e.target.value)
                  }))}
                  step="0.1"
                  min="0.1"
                  max="0.5"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800"
                />
              </div>
            </div>
          )}

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">Training Progress:</span>
              <span className="text-sm font-medium">
                {currentEpoch} / {trainingConfig.epochs} epochs
                {trainingComplete && ' (Complete)'}
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
              <div
                className="bg-blue-600 dark:bg-blue-500 h-2.5 rounded-full transition-all"
                style={{ width: `${(currentEpoch / trainingConfig.epochs) * 100}%` }}
              ></div>
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={isTraining ? stopTraining : startTraining}
                disabled={trainingVideos.length === 0 || Array.from(videoStatus.values()).some(s => s.isProcessing)}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition ${
                  isTraining
                    ? ' bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {isTraining ? (
                  <>
                    <X size={18} />
                    Stop Training
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    Start Training
                  </>
                )}
              </button>
              <button
                onClick={() => {
                  setMetrics([]);
                  setCurrentEpoch(0);
                  setTrainingComplete(false);
                }}
                className="px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
              >
                <RefreshCw size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Training Metrics Chart */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-800">
        <h3 className="text-lg font-semibold mb-4">Training Metrics</h3>
        {metrics.length > 0 ? (
          <div className="h-[400px]">
            <Line
              data={chartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  y: {
                    beginAtZero: true,
                    grid: {
                      color: 'rgba(0, 0, 0, 0.1)'
                    }
                  },
                  x: {
                    grid: {
                      display: false
                    }
                  }
                },
                plugins: {
                  legend: {
                    position: 'top' as const,
                  },
                  tooltip: {
                    mode: 'index' as const,
                    intersect: false,
                  }
                }
              }}
            />
          </div>
        ) : (
          <div className="h-[400px] flex items-center justify-center text-gray-400 dark:text-gray-600">
            <p>No training data available</p>
          </div>
        )}
      </div>
    </div>
  );
}