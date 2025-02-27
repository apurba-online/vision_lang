import React, { useEffect, useRef, useState } from 'react';
import { Camera, CameraOff, FlipHorizontal, Play, Pause, Settings, AlertCircle } from 'lucide-react';
import Webcam from 'react-webcam';

declare const faceapi: any;

// Constants
const MIN_SCORE = 0.3;
const MAX_RESULTS = 5;

interface FaceDetection {
  detection: {
    box: { x: number; y: number; width: number; height: number };
  };
  landmarks: { positions: Array<{ x: number; y: number }> };
  expressions: { [key: string]: number };
  age: number;
  gender: string;
  genderProbability: number;
}

export function FaceAnalysisTab() {
  const [isCameraEnabled, setIsCameraEnabled] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [error, setError] = useState<string | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [fps, setFps] = useState<number>(0);
  const [showSettings, setShowSettings] = useState(false);
  const [detectionSettings, setDetectionSettings] = useState({
    minScore: MIN_SCORE,
    maxResults: MAX_RESULTS
  });
  
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const lastFrameTime = useRef<number>(0);

  useEffect(() => {
    const loadModels = async () => {
      try {
        // Wait for faceapi to be available from CDN
        if (typeof faceapi === 'undefined') {
          throw new Error('Face API not loaded. Please check your internet connection.');
        }

        // Load models from CDN
        const modelPath = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(modelPath),
          faceapi.nets.faceLandmark68Net.loadFromUri(modelPath),
          faceapi.nets.faceExpressionNet.loadFromUri(modelPath),
          faceapi.nets.ageGenderNet.loadFromUri(modelPath)
        ]);

        console.log('Face-API models loaded successfully');
        setIsModelLoaded(true);
        setError(null);
      } catch (err) {
        console.error('Error loading models:', err);
        setError('Failed to load face detection models. Please check your internet connection.');
      }
    };

    loadModels();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const toggleCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
  };

  const toggleCameraEnabled = () => {
    setIsCameraEnabled(prev => !prev);
    if (isAnalyzing) {
      setIsAnalyzing(false);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
  };

  const drawFaces = (canvas: HTMLCanvasElement, detections: FaceDetection[], currentFps: number) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw FPS counter
    ctx.font = 'small-caps 20px "Segoe UI"';
    ctx.fillStyle = 'white';
    ctx.fillText(`FPS: ${currentFps.toFixed(1)}`, 10, 25);

    for (const detection of detections) {
      // Draw face box
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'deepskyblue';
      ctx.fillStyle = 'deepskyblue';
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.rect(
        detection.detection.box.x,
        detection.detection.box.y,
        detection.detection.box.width,
        detection.detection.box.height
      );
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Get dominant expression
      const expression = Object.entries(detection.expressions)
        .sort((a, b) => b[1] - a[1])[0];

      // Draw text background
      const textY = detection.detection.box.y - 60;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(
        detection.detection.box.x,
        textY,
        200,
        55
      );

      // Draw text with shadow effect
      const textX = detection.detection.box.x + 5;
      ctx.font = '16px "Segoe UI"';
      
      // Text shadow
      ctx.fillStyle = 'black';
      [
        `Gender: ${Math.round(100 * detection.genderProbability)}% ${detection.gender}`,
        `Expression: ${Math.round(100 * expression[1])}% ${expression[0]}`,
        `Age: ${Math.round(detection.age)} years`
      ].forEach((text, i) => {
        ctx.fillText(text, textX + 1, textY + 18 + (i * 18) + 1);
      });

      // Main text
      ctx.fillStyle = 'white';
      [
        `Gender: ${Math.round(100 * detection.genderProbability)}% ${detection.gender}`,
        `Expression: ${Math.round(100 * expression[1])}% ${expression[0]}`,
        `Age: ${Math.round(detection.age)} years`
      ].forEach((text, i) => {
        ctx.fillText(text, textX, textY + 18 + (i * 18));
      });

      // Draw face landmarks
      ctx.fillStyle = '#00ff00';
      for (const point of detection.landmarks.positions) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  };

  const detectFaces = async () => {
    if (!isAnalyzing || !webcamRef.current?.video || !canvasRef.current) return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;

    // Ensure canvas dimensions match video
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Calculate FPS
    const now = performance.now();
    const elapsed = now - lastFrameTime.current;
    const currentFps = 1000 / elapsed;
    lastFrameTime.current = now;
    setFps(currentFps);

    try {
      // Detect faces with all features
      const detections = await faceapi
        .detectAllFaces(
          video,
          new faceapi.SsdMobilenetv1Options({
            minConfidence: detectionSettings.minScore,
            maxResults: detectionSettings.maxResults
          })
        )
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender();

      // Draw results
      drawFaces(canvas, detections, currentFps);
    } catch (err) {
      console.error('Detection error:', err);
    }

    // Continue detection loop
    animationRef.current = requestAnimationFrame(detectFaces);
  };

  const toggleAnalysis = () => {
    if (!isAnalyzing) {
      setIsAnalyzing(true);
      lastFrameTime.current = performance.now();
      detectFaces();
    } else {
      setIsAnalyzing(false);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
  };

  if (!isModelLoaded) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-lg">Loading Face Detection Models...</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">This may take a moment</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="grid lg:grid-cols-5 gap-6">
        {/* Video Section */}
        <div className="lg:col-span-3 bg-white dark:bg-gray-900 rounded-2xl shadow-xl overflow-hidden border border-gray-200 dark:border-gray-800">
          <div className="p-4 h-full flex flex-col">
            <div className="relative flex-1 rounded-xl overflow-hidden">
              <div className="h-full relative rounded-xl overflow-hidden">
                {isCameraEnabled ? (
                  <>
                    <Webcam
                      ref={webcamRef}
                      audio={false}
                      className="w-full h-full object-cover"
                      screenshotFormat="image/jpeg"
                      videoConstraints={{
                        facingMode,
                        width: { ideal: 1920 },
                        height: { ideal: 1080 },
                        aspectRatio: 16/9
                      }}
                    />
                    <canvas
                      ref={canvasRef}
                      className="absolute inset-0 w-full h-full"
                    />
                    {isAnalyzing && (
                      <div className="absolute top-4 left-4 bg-black/50 backdrop-blur-sm rounded-lg px-3 py-1.5 text-white text-sm">
                        FPS: {fps.toFixed(1)}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gray-900">
                    <CameraOff size={48} className="text-gray-600" />
                  </div>
                )}
              </div>
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
                      onClick={toggleAnalysis}
                      className={`p-3 rounded-full transition backdrop-blur-sm ${
                        isAnalyzing 
                          ? 'bg-red-600 hover:bg-red-700' 
                          : 'bg-blue-600 hover:bg-blue-700'
                      } text-white`}
                    >
                      {isAnalyzing ? <Pause size={20} /> : <Play size={20} />}
                    </button>
                    <button
                      onClick={() => setShowSettings(!showSettings)}
                      className={`p-3 rounded-full transition backdrop-blur-sm ${
                        showSettings
                          ? 'bg-blue-600 hover:bg-blue-700'
                          : 'bg-gray-900/80 hover:bg-gray-800'
                      } text-white`}
                    >
                      <Settings size={20} />
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Info Section */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-900 rounded-2xl shadow-xl overflow-hidden border border-gray-200 dark:border-gray-800">
          <div className="p-6">
            {showSettings ? (
              <>
                <h3 className="text-lg font-semibold mb-4">Detection Settings</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Minimum Detection Score
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.1"
                      value={detectionSettings.minScore}
                      onChange={(e) => setDetectionSettings(prev => ({
                        ...prev,
                        minScore: parseFloat(e.target.value)
                      }))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>0.1</span>
                      <span>{detectionSettings.minScore}</span>
                      <span>0.9</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Maximum Results
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="1"
                      value={detectionSettings.maxResults}
                      onChange={(e) => setDetectionSettings(prev => ({
                        ...prev,
                        maxResults: parseInt(e.target.value)
                      }))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>1</span>
                      <span>{detectionSettings.maxResults}</span>
                      <span>10</span>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                <h3 className="text-lg font-semibold mb-4">Face Analysis Features</h3>
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">Available Detection</h4>
                    <ul className="list-disc list-inside space-y-2 text-sm text-blue-600 dark:text-blue-400">
                      <li>Face Detection</li>
                      <li>68 Point Face Landmarks</li>
                      <li>Face Expressions</li>
                      <li>Age Estimation</li>
                      <li>Gender Recognition</li>
                    </ul>
                  </div>
                  
                  <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <h4 className="font-medium mb-2">Instructions</h4>
                    <ol className="list-decimal list-inside space-y-2 text-sm text-gray-600 dark:text-gray-400">
                      <li>Enable camera access</li>
                      <li>Position your face in the frame</li>
                      <li>Click play to start analysis</li>
                      <li>Results will be displayed in real-time</li>
                    </ol>
                  </div>

                  {error && (
                    <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center gap-2 text-red-600 dark:text-red-400 text-sm">
                      <AlertCircle size={16} className="flex-shrink-0" />
                      <p>{error}</p>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}