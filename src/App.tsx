import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import Webcam from 'react-webcam';
import { 
  Upload, Camera, Play, Pause, MessageSquare, Loader2, 
  Menu, X, Video, Sun, Moon, Upload as UploadIcon,
  Settings, Info, Github, FlipHorizontal
} from 'lucide-react';
import { loadModel, analyzeVideo, analyzeFrame, type PersonAnnotation } from './utils/model';

type AnalysisMode = 'upload' | 'camera';
type VideoSource = string | null;
type Theme = 'light' | 'dark';

function App() {
  const [mode, setMode] = useState<AnalysisMode>('camera');
  const [videoSource, setVideoSource] = useState<VideoSource>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [analysis, setAnalysis] = useState<string>('');
  const [annotations, setAnnotations] = useState<PersonAnnotation[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');
  const [isMobile, setIsMobile] = useState(false);
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window !== 'undefined') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'dark';
  });
  
  const analysisLoopRef = useRef<number>();
  const isAnalyzingRef = useRef(false);
  const lastAnalysisTimeRef = useRef<number>(0);
  const analysisUpdateIntervalRef = useRef<NodeJS.Timeout>();
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const webcamRef = React.useRef<Webcam>(null);

  useEffect(() => {
    // Check if device is mobile
    const checkMobile = () => {
      setIsMobile(/iPhone|iPad|iPod|Android/i.test(navigator.userAgent));
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  const toggleCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
  };

  useEffect(() => {
    loadModel().then(() => {
      setIsModelLoaded(true);
    }).catch(error => {
      setError('Failed to load the AI model. Please refresh the page.');
      console.error('Model loading error:', error);
    });

    return () => {
      if (analysisLoopRef.current) {
        cancelAnimationFrame(analysisLoopRef.current);
      }
      if (analysisUpdateIntervalRef.current) {
        clearInterval(analysisUpdateIntervalRef.current);
      }
    };
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      if (file.size > 100 * 1024 * 1024) {
        setError('Video file size must be less than 100MB');
        return;
      }
      const videoUrl = URL.createObjectURL(file);
      setVideoSource(videoUrl);
      setVideoFile(file);
      setError('');
      setMode('upload');
      setIsSidebarOpen(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.webm', '.ogg']
    },
    maxFiles: 1
  });

  const handleQuestionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    if (!videoFile) {
      setError('Please upload a video first.');
      return;
    }

    setIsLoading(true);
    try {
      const result = await analyzeVideo(videoFile);
      setAnalysis(result);
    } catch (error) {
      setError('Error analyzing video. Please try again.');
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeWebcamFrame = async () => {
    if (!webcamRef.current?.video || !isAnalyzingRef.current) return;

    const currentTime = Date.now();
    if (currentTime - lastAnalysisTimeRef.current < 10000) {
      analysisLoopRef.current = requestAnimationFrame(analyzeWebcamFrame);
      return;
    }

    try {
      const result = await analyzeFrame(webcamRef.current.video);
      setAnalysis(result.commentary);
      setAnnotations(result.annotations || []);
      lastAnalysisTimeRef.current = currentTime;
    } catch (error) {
      console.error('Analysis error:', error);
    }

    if (isAnalyzingRef.current) {
      analysisLoopRef.current = requestAnimationFrame(analyzeWebcamFrame);
    }
  };

  const toggleRecording = () => {
    if (!isRecording) {
      setIsRecording(true);
      isAnalyzingRef.current = true;
      lastAnalysisTimeRef.current = 0;
      analyzeWebcamFrame();
    } else {
      setIsRecording(false);
      isAnalyzingRef.current = false;
      if (analysisLoopRef.current) {
        cancelAnimationFrame(analysisLoopRef.current);
      }
      setAnalysis('');
    }
  };

  useEffect(() => {
    setAnalysis('');
    setAnnotations([]);
    setIsRecording(false);
    isAnalyzingRef.current = false;
    if (analysisLoopRef.current) {
      cancelAnimationFrame(analysisLoopRef.current);
    }
  }, [mode]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  if (!isModelLoaded) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-900 dark:to-gray-800 text-gray-900 dark:text-white flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4" />
          <p className="text-xl">Loading AI Model...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-900 dark:to-gray-800 text-gray-900 dark:text-white flex flex-col">
      {/* Sidebar Menu */}
      <div 
        className={`fixed inset-y-0 left-0 w-full md:w-80 bg-white dark:bg-gray-800 transform transition-transform duration-300 ease-in-out z-50 ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } shadow-xl`}
      >
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <Video className="w-6 h-6 text-blue-600 dark:text-blue-500" />
                <h2 className="text-xl font-semibold">Video Analysis AI</h2>
              </div>
              <button
                onClick={() => setIsSidebarOpen(false)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition"
              >
                <X size={24} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto">
            <div className="p-4">
              <div className="space-y-4">
                {/* Upload Section */}
                <div className="mb-8">
                  <h3 className="text-lg font-medium mb-4">Upload Video</h3>
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${
                      isDragActive 
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                        : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                    }`}
                  >
                    <input {...getInputProps()} />
                    <UploadIcon className="mx-auto mb-4 text-gray-400 dark:text-gray-500" size={48} />
                    <p className="text-sm mb-2">
                      {isDragActive
                        ? 'Drop the video here'
                        : 'Drag & drop a video file here, or click to select'}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Supports MP4, WebM, and OGG formats (max 100MB)
                    </p>
                  </div>
                </div>

                {/* Menu Items */}
                <div className="space-y-2">
                  <button className="w-full flex items-center gap-3 px-4 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition">
                    <Settings size={20} />
                    <span>Settings</span>
                  </button>
                  <button className="w-full flex items-center gap-3 px-4 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition">
                    <Info size={20} />
                    <span>About</span>
                  </button>
                  <button className="w-full flex items-center gap-3 px-4 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition">
                    <Github size={20} />
                    <span>GitHub</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Theme Toggle in Sidebar Footer */}
          <div className="border-t border-gray-200 dark:border-gray-700 p-4">
            <button
              onClick={toggleTheme}
              className="w-full flex items-center gap-3 px-4 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
            >
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
              <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Fixed Header */}
      <header className="fixed top-0 left-0 right-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm z-40 border-b border-gray-200 dark:border-gray-700 h-16">
        <div className="max-w-7xl mx-auto px-4 md:px-8 h-full flex justify-between items-center">
          <div className="flex items-center gap-3">
            <Video className="w-6 h-6 text-blue-600 dark:text-blue-500" />
            <h1 className="text-xl md:text-2xl font-bold">Video Analysis AI</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={toggleTheme}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition"
            >
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition"
            >
              <Menu size={20} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 mt-16 px-4 md:px-8 py-6">
        <div className="max-w-7xl mx-auto">
          {error && (
            <div className="mb-4 p-4 bg-red-100 dark:bg-red-500/20 border border-red-200 dark:border-red-500 rounded-lg text-red-800 dark:text-red-200 text-sm md:text-base">
              {error}
            </div>
          )}

          {/* Responsive Layout Container */}
          <div className="flex flex-col lg:flex-row gap-6 md:gap-8 min-h-[calc(100vh-7rem)]">
            {/* Video Section */}
            <div className="lg:w-3/5 lg:h-[calc(100vh-7rem)]">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl h-full">
                <div className="p-4 md:p-6 h-full flex flex-col">
                  <div ref={videoContainerRef} className="relative flex-1">
                    {mode === 'upload' && videoSource ? (
                      <div className="h-full flex flex-col">
                        <video
                          src={videoSource}
                          controls
                          className="w-full h-full object-contain rounded-lg mb-4"
                        />
                        <button
                          onClick={handleQuestionSubmit}
                          disabled={isLoading}
                          className="w-full flex items-center justify-center gap-2 px-4 md:px-6 py-2 md:py-3 bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed text-sm md:text-base"
                        >
                          {isLoading ? (
                            <Loader2 size={20} className="animate-spin" />
                          ) : (
                            <MessageSquare size={20} />
                          )}
                          Analyze Video
                        </button>
                      </div>
                    ) : (
                      <div className="h-full relative">
                        <Webcam
                          ref={webcamRef}
                          audio={false}
                          className="w-full h-full object-contain rounded-lg"
                          screenshotFormat="image/jpeg"
                          videoConstraints={{
                            facingMode,
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                          }}
                        />
                        <div className="absolute bottom-4 right-4 flex gap-2">
                          <button
                            onClick={toggleCamera}
                            className="p-2 md:p-3 rounded-full bg-gray-800/80 hover:bg-gray-700 text-white transition backdrop-blur-sm"
                            title="Switch Camera"
                          >
                            <FlipHorizontal size={24} />
                          </button>
                          <button
                            onClick={toggleRecording}
                            className={`p-2 md:p-3 rounded-full transition ${
                              isRecording 
                                ? 'bg-red-600 hover:bg-red-700' 
                                : 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600'
                            } text-white`}
                          >
                            {isRecording ? <Pause size={24} /> : <Play size={24} />}
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Annotation Boxes */}
                    {annotations.map((annotation, index) => {
                      const [x, y, width, height] = annotation.bbox;
                      const containerRect = videoContainerRef.current?.getBoundingClientRect();
                      if (!containerRect) return null;

                      const scaleX = containerRect.width / (webcamRef.current?.video?.videoWidth || 1);
                      const scaleY = containerRect.height / (webcamRef.current?.video?.videoHeight || 1);

                      return (
                        <div
                          key={index}
                          className="absolute pointer-events-none"
                          style={{
                            left: `${x * scaleX}px`,
                            top: `${y * scaleY}px`,
                            width: `${width * scaleX}px`,
                            height: `${height * scaleY}px`,
                            border: '2px solid #3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)'
                          }}
                        >
                          <div className="absolute -top-6 md:-top-8 left-0 bg-blue-600 dark:bg-blue-500 text-white px-1.5 md:px-2 py-0.5 md:py-1 rounded text-xs whitespace-nowrap">
                            {annotation.class ? (
                              annotation.class
                            ) : (
                              `${annotation.gender} • ${annotation.ageRange} • ${annotation.expression}`
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>

            {/* Analysis Section */}
            {analysis && (
              <div className="lg:w-2/5 lg:h-[calc(100vh-7rem)]">
                <div className="bg-white dark:bg-gray-800 p-4 md:p-6 rounded-xl shadow-xl h-full flex flex-col">
                  <h3 className="text-base md:text-lg font-semibold mb-3 md:mb-4">Analysis Results</h3>
                  <div className="bg-gray-50 dark:bg-gray-700 p-3 md:p-4 rounded-lg flex-1 overflow-y-auto text-sm md:text-base">
                    <p className="whitespace-pre-wrap">{analysis}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
