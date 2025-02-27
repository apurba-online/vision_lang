import React, { useState, useRef } from 'react';
import { Upload, Play, AlertCircle, Video, X, Gauge } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { loadTrainedModel } from '../utils/train';
import { processVideoForPrediction } from '../utils/dataCollection';

const SUPPORTED_VIDEO_FORMATS = {
  'video/mp4': ['.mp4'],
  'video/webm': ['.webm'],
  'video/x-matroska': ['.mkv'],
  'video/quicktime': ['.mov'],
  'video/x-msvideo': ['.avi']
};

interface PredictionResult {
  timestamp: number;
  isViolent: boolean;
  confidence: number;
  detections: Array<{
    bbox: number[];
    label: string;
    score: number;
  }>;
}

export function TestingTab() {
  const [testVideo, setTestVideo] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const onDrop = async (acceptedFiles: File[]) => {
    setError(null);
    const file = acceptedFiles[0];
    if (file) {
      if (file.size > 100 * 1024 * 1024) {
        setError('File is too large. Maximum size is 100MB');
        return;
      }
      setTestVideo(file);
      setPredictions([]);
      setCurrentPrediction(null);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: SUPPORTED_VIDEO_FORMATS,
    maxSize: 100 * 1024 * 1024,
    multiple: false
  });

  const startAnalysis = async () => {
    if (!testVideo || !videoRef.current) return;

    setIsAnalyzing(true);
    setError(null);
    setPredictions([]);

    try {
      const model = await loadTrainedModel();
      if (!model) {
        throw new Error('No trained model found. Please train the model first.');
      }

      videoRef.current.src = URL.createObjectURL(testVideo);
      await videoRef.current.play();

      const results = await processVideoForPrediction(
        videoRef.current,
        model,
        (prediction) => {
          setPredictions(prev => [...prev, prediction]);
          setCurrentPrediction(prediction);
          drawDetections(prediction);
        }
      );

      console.log('Analysis complete:', results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const drawDetections = (prediction: PredictionResult) => {
    if (!canvasRef.current || !videoRef.current) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

    // Set canvas dimensions to match video
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;

    // Draw bounding boxes
    prediction.detections.forEach(detection => {
      const [x, y, width, height] = detection.bbox;
      
      // Set style based on confidence and violence prediction
      ctx.strokeStyle = prediction.isViolent ? 'red' : 'green';
      ctx.lineWidth = 2;
      
      // Draw rectangle
      ctx.strokeRect(x, y, width, height);
      
      // Draw label
      ctx.fillStyle = prediction.isViolent ? 'red' : 'green';
      ctx.font = '16px Arial';
      ctx.fillText(
        `${detection.label} (${Math.round(detection.score * 100)}%)`,
        x,
        y - 5
      );
    });

    // Draw overall prediction
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, 10, 200, 60);
    ctx.fillStyle = prediction.isViolent ? 'red' : 'green';
    ctx.font = 'bold 16px Arial';
    ctx.fillText(
      `Violence: ${prediction.isViolent ? 'Detected' : 'Not Detected'}`,
      20,
      35
    );
    ctx.fillText(
      `Confidence: ${Math.round(prediction.confidence * 100)}%`,
      20,
      55
    );
  };

  return (
    <div className="p-6 space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Upload Section */}
        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-800">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Video className="w-5 h-5 text-blue-500" />
            Test Video
          </h3>
          
          {!testVideo ? (
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
                  ? 'Drop the video here'
                  : 'Drag & drop a video file here, or click to select'}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Supports MP4, WebM, MOV, MKV, and AVI formats (max 100MB)
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="flex items-center gap-2">
                  <Video className="w-4 h-4 text-blue-500" />
                  <span className="text-sm truncate">{testVideo.name}</span>
                </div>
                <button
                  onClick={() => {
                    setTestVideo(null);
                    setPredictions([]);
                    setCurrentPrediction(null);
                  }}
                  className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-full"
                >
                  <X size={16} />
                </button>
              </div>

              <button
                onClick={startAnalysis}
                disabled={isAnalyzing}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? (
                  <>
                    <Gauge className="animate-spin" size={18} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    Start Analysis
                  </>
                )}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-4 p-3 bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400 rounded-lg text-sm flex items-center gap-2">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-800">
          <h3 className="text-lg font-semibold mb-4">Analysis Results</h3>
          
          {predictions.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">
                  Analyzed Frames: {predictions.length}
                </span>
                <span className="text-sm font-medium">
                  Violence Detected: {predictions.filter(p => p.isViolent).length} frames
                </span>
              </div>

              <div className="h-[200px] overflow-y-auto scrollbar-thin">
                <div className="space-y-2">
                  {predictions.map((prediction, index) => (
                    <div
                      key={index}
                      className={`p-3 rounded-lg ${
                        prediction.isViolent
                          ? 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400'
                          : 'bg-green-50 dark:bg-green-900/20 text-green-600 dark:text-green-400'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">
                          Frame {index + 1} ({(prediction.timestamp / 1000).toFixed(2)}s)
                        </span>
                        <span className="text-sm">
                          Confidence: {Math.round(prediction.confidence * 100)}%
                        </span>
                      </div>
                      <div className="text-xs mt-1">
                        Detections: {prediction.detections.length} person(s)
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-[200px] flex items-center justify-center text-gray-400 dark:text-gray-600">
              <p className="text-sm">No analysis results available</p>
            </div>
          )}
        </div>
      </div>

      {/* Video Preview */}
      <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-800">
        <h3 className="text-lg font-semibold mb-4">Video Preview</h3>
        <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
          <video
            ref={videoRef}
            className="absolute inset-0 w-full h-full"
            controls
            crossOrigin="anonymous"
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
          />
        </div>
      </div>
    </div>
  );
}