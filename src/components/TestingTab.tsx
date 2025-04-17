import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, AlertCircle, Video, X, Gauge, Camera, CameraOff, FlipHorizontal, Pause } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
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
  const [mode, setMode] = useState<'upload' | 'webcam'>('upload');
  const [isCameraEnabled, setIsCameraEnabled] = useState(false);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [fps, setFps] = useState<number>(0);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [webcamReady, setWebcamReady] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<tf.LayersModel | null>(null);
  const cocoModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const fpsIntervalRef = useRef<number | null>(null);
  const frameCountRef = useRef<number>(0);

  useEffect(() => {
    const initTf = async () => {
      try {
        await tf.ready();
        console.log('TensorFlow.js initialized successfully');
      } catch (err) {
        console.error('Error initializing TensorFlow.js:', err);
        setError('Failed to initialize TensorFlow.js. Please check your browser compatibility.');
      }
    };
    
    initTf();
    
    return () => {
      if (cleanupRef.current) {
        cleanupRef.current();
      }
      if (fpsIntervalRef.current) {
        clearInterval(fpsIntervalRef.current);
      }
    };
  }, []);

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

      cleanupRef.current = await processVideoForPrediction(
        videoRef.current,
        model,
        (prediction) => {
          setPredictions(prev => [...prev.slice(-99), prediction]);
          setCurrentPrediction(prediction);
          drawDetections(prediction);
          frameCountRef.current++;
        }
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
      setIsAnalyzing(false);
    }
  };

  const toggleCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
    setWebcamReady(false);
  };

  const toggleCameraEnabled = () => {
    setIsCameraEnabled(prev => !prev);
    if (isAnalyzing) {
      stopWebcamAnalysis();
    }
    setWebcamReady(false);
  };

  const handleWebcamReady = () => {
    console.log('Webcam is ready');
    setWebcamReady(true);
  };

  const loadModels = async () => {
    setIsModelLoading(true);
    setError(null);
    
    try {
      if (!modelRef.current) {
        const model = await loadTrainedModel();
        if (!model) {
          throw new Error('No trained model found. Please train the model first.');
        }
        modelRef.current = model;
      }
      
      if (!cocoModelRef.current) {
        const cocoModel = await cocoSsd.load({
          base: 'lite_mobilenet_v2'
        });
        cocoModelRef.current = cocoModel;
      }
      
      return true;
    } catch (err) {
      console.error('Error loading models:', err);
      setError(err instanceof Error ? err.message : 'Failed to load models');
      return false;
    } finally {
      setIsModelLoading(false);
    }
  };

  const startWebcamAnalysis = async () => {
    if (!webcamRef.current?.video || isAnalyzing || !webcamReady) {
      if (!webcamReady) {
        setError('Webcam is not ready yet. Please wait a moment and try again.');
      }
      return;
    }

    try {
      const modelsLoaded = await loadModels();
      if (!modelsLoaded) {
        throw new Error('Failed to load required models');
      }

      setIsAnalyzing(true);
      setError(null);
      setPredictions([]);
      frameCountRef.current = 0;

      // Start FPS counter
      if (fpsIntervalRef.current) {
        clearInterval(fpsIntervalRef.current);
      }
      fpsIntervalRef.current = window.setInterval(() => {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
      }, 1000);

      cleanupRef.current = await processVideoForPrediction(
        webcamRef.current.video,
        modelRef.current!,
        (prediction) => {
          setPredictions(prev => [...prev.slice(-99), prediction]);
          setCurrentPrediction(prediction);
          drawDetections(prediction);
          frameCountRef.current++;
        }
      );
    } catch (err) {
      console.error('Error starting webcam analysis:', err);
      setError(err instanceof Error ? err.message : 'An error occurred during analysis');
      stopWebcamAnalysis();
    }
  };
  
  const stopWebcamAnalysis = () => {
    setIsAnalyzing(false);
    if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }
    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
      fpsIntervalRef.current = null;
    }
  };

  const drawDetections = (prediction: PredictionResult) => {
    if (!canvasRef.current) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    const videoElement = mode === 'upload' ? videoRef.current : webcamRef.current?.video;
    if (!videoElement) return;

    // Get container dimensions
    const containerElement = canvasRef.current.parentElement;
    if (!containerElement) return;

    const containerWidth = containerElement.clientWidth;
    const containerHeight = containerElement.clientHeight;

    // Set canvas dimensions to match container
    canvasRef.current.width = containerWidth;
    canvasRef.current.height = containerHeight;

    // Calculate scale factors
    const scaleX = containerWidth / videoElement.videoWidth;
    const scaleY = containerHeight / videoElement.videoHeight;
    const scale = Math.min(scaleX, scaleY);

    // Calculate centered position
    const offsetX = (containerWidth - (videoElement.videoWidth * scale)) / 2;
    const offsetY = (containerHeight - (videoElement.videoHeight * scale)) / 2;

    // Clear previous drawings
    ctx.clearRect(0, 0, containerWidth, containerHeight);

    // Draw FPS counter
    if (isAnalyzing) {
      ctx.font = 'bold 16px Arial';
      ctx.fillStyle = 'white';
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 3;
      const fpsText = `FPS: ${fps}`;
      ctx.strokeText(fpsText, 10, 25);
      ctx.fillText(fpsText, 10, 25);
    }

    // Draw bounding boxes
    prediction.detections.forEach(detection => {
      const [x, y, width, height] = detection.bbox;
      
      // Scale and offset coordinates
      const scaledX = (x * scale) + offsetX;
      const scaledY = (y * scale) + offsetY;
      const scaledWidth = width * scale;
      const scaledHeight = height * scale;
      
      // Draw box with shadow effect
      ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
      ctx.shadowBlur = 5;
      ctx.lineWidth = 2;
      ctx.strokeStyle = prediction.isViolent ? 'red' : 'green';
      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
      
      // Reset shadow for text
      ctx.shadowColor = 'transparent';
      
      // Draw background for label
      const label = `${detection.label} (${Math.round(detection.score * 100)}%)`;
      const labelWidth = ctx.measureText(label).width + 10;
      ctx.fillStyle = prediction.isViolent ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 128, 0, 0.8)';
      ctx.fillRect(scaledX, scaledY - 25, labelWidth, 20);
      
      // Draw label text
      ctx.fillStyle = 'white';
      ctx.font = '14px Arial';
      ctx.fillText(label, scaledX + 5, scaledY - 10);
    });

    // Draw overall prediction
    if (prediction.detections.length > 0) {
      const boxWidth = 200;
      const boxHeight = 60;
      const margin = 10;
      
      // Draw background box
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(
        containerWidth - boxWidth - margin,
        margin,
        boxWidth,
        boxHeight
      );
      
      // Draw prediction text
      ctx.fillStyle = prediction.isViolent ? 'red' : 'green';
      ctx.font = 'bold 16px Arial';
      ctx.fillText(
        `Violence: ${prediction.isViolent ? 'Detected' : 'Not Detected'}`,
        containerWidth - boxWidth - margin + 10,
        margin + 25
      );
      ctx.fillText(
        `Confidence: ${Math.round(prediction.confidence * 100)}%`,
        containerWidth - boxWidth - margin + 10,
        margin + 45
      );
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Mode Toggle */}
      <div className="flex justify-center mb-4">
        <div className="inline-flex rounded-md shadow-sm" role="group">
          <button
            type="button"
            onClick={() => {
              setMode('upload');
              stopWebcamAnalysis();
            }}
            className={`px-4 py-2 text-sm font-medium rounded-l-lg ${
              mode === 'upload'
                ? 'bg-blue-600 text-white'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            <div className="flex items-center gap-2">
              <Upload size={16} />
              <span>Upload Video</span>
            </div>
          </button>
          <button
            type="button"
            onClick={() => {
              setMode('webcam');
              if (testVideo) {
                URL.revokeObjectURL(videoRef.current?.src || '');
                setTestVideo(null);
              }
            }}
            className={`px-4 py-2 text-sm font-medium rounded-r-lg ${
              mode === 'webcam'
                ? 'bg-blue-600 text-white'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            <div className="flex items-center gap-2">
              <Camera size={16} />
              <span>Live Camera</span>
            </div>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Upload/Webcam Section */}
        <div className="bg-white dark:bg-gray-900 rounded-xl p-6 shadow-lg border border-gray-200 dark:border-gray-800">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            {mode === 'upload' ? (
              <>
                <Video className="w-5 h-5 text-blue-500" />
                Test Video
              </>
            ) : (
              <>
                <Camera className="w-5 h-5 text-blue-500" />
                Live Camera
              </>
            )}
          </h3>
          
          {mode === 'upload' ? (
            !testVideo ? (
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
            )
          ) : (
            <div className="space-y-4">
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                {isCameraEnabled ? (
                  <>
                    <Webcam
                      ref={webcamRef}
                      audio={false}
                      className="w-full h-full object-cover"
                      screenshotFormat="image/jpeg"
                      videoConstraints={{
                        facingMode,
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                      }}
                      onUserMedia={handleWebcamReady}
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute inset-0 w-full h-full pointer-events-none"
                    />
                    {isAnalyzing && (
                      <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm rounded-lg px-3 py-1.5 text-white text-sm">
                        FPS: {fps}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <CameraOff size={48} className="text-gray-600" />
                  </div>
                )}
                
                {/* Camera controls */}
                <div className="absolute bottom-4 right-4 flex gap-2">
                  <button
                    onClick={toggleCameraEnabled}
                    className="p-3 rounded-full bg-gray-900/80 hover:bg-gray-800 text-white transition backdrop-blur-sm"
                    title={isCameraEnabled ? "Turn Off Camera" : "Turn On Camera"}
                  >
                    {isCameraEnabled ? <CameraOff size={20} /> : <Camera size={20} />}
                  </button>
                  {isCameraEnabled && (
                    <>
                      <button
                        onClick={toggleCamera}
                        className="p-3 rounded-full bg-gray-900/80 hover:bg-gray-800 text-white transition backdrop-blur-sm"
                        title="Switch Camera"
                      >
                        <FlipHorizontal size={20} />
                      </button>
                      <button
                        onClick={isAnalyzing ? stopWebcamAnalysis : startWebcamAnalysis}
                        disabled={isModelLoading || !webcamReady}
                        className={`p-3 rounded-full transition backdrop-blur-sm ${
                          isAnalyzing 
                            ? 'bg-red-600 hover:bg-red-700' 
                            : 'bg-blue-600 hover:bg-blue-700'
                        } text-white disabled:opacity-50 disabled:cursor-not-allowed`}
                      >
                        {isModelLoading ? (
                          <Gauge className="animate-spin" size={20} />
                        ) : isAnalyzing ? (
                          <Pause size={20} />
                        ) : (
                          <Play size={20} />
                        )}
                      </button>
                    </>
                  )}
                </div>
              </div>
              
              {isCameraEnabled && !isAnalyzing && (
                <button
                  onClick={startWebcamAnalysis}
                  disabled={isModelLoading || !webcamReady}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isModelLoading ? (
                    <>
                      <Gauge className="animate-spin" size={18} />
                      Loading Models...
                    </>
                  ) : !webcamReady ? (
                    <>
                      <Gauge className="animate-spin" size={18} />
                      Initializing Camera...
                    </>
                  ) : (
                    <>
                      <Play size={18} />
                      Start Live Analysis
                    </>
                  )}
                </button>
              )}
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
                          Frame {index + 1} {mode === 'upload' ? `(${(prediction.timestamp / 1000).toFixed(2)}s)` : ''}
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

      {/* Video Preview - Only show for upload mode */}
      {mode === 'upload' && (
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
      )}
    </div>
  );
}