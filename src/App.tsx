import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import Webcam from 'react-webcam';
import { 
  Upload, Camera, Play, Pause, MessageSquare, Loader2, 
  Menu, X, Video, Sun, Moon, Upload as UploadIcon,
  Settings, Info, Github, FlipHorizontal, Bug, Terminal
} from 'lucide-react';
import { loadModel, analyzeVideo, analyzeFrame, type PersonAnnotation } from './utils/model';

type AnalysisMode = 'upload' | 'camera';
type VideoSource = string | null;
type Theme = 'light' | 'dark';

type DebugInfo = {
  hasFrame: boolean;
  frameSize: number;
  messageCount: number;
  lastUpdate: string;
};

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
  const [showDebug, setShowDebug] = useState(false);
  const [debugInfo, setDebugInfo] = useState<DebugInfo>({
    hasFrame: false,
    frameSize: 0,
    messageCount: 0,
    lastUpdate: new Date().toISOString()
  });
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

  const clearAnalysis = useCallback(() => {
    setAnalysis('');
    setAnnotations([]);
    if (analysisLoopRef.current) {
      cancelAnimationFrame(analysisLoopRef.current);
      analysisLoopRef.current = undefined;
    }
    isAnalyzingRef.current = false;
    lastAnalysisTimeRef.current = 0;
  }, []);

  useEffect(() => {
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
    if (!webcamRef.current?.video || !isAnalyzingRef.current) {
      clearAnalysis();
      return;
    }

    const currentTime = Date.now();
    if (currentTime - lastAnalysisTimeRef.current < 10000) {
      analysisLoopRef.current = requestAnimationFrame(analyzeWebcamFrame);
      return;
    }

    try {
      const result = await analyzeFrame(webcamRef.current.video);
      if (isAnalyzingRef.current) { // Only update if still recording
        setAnalysis(result.commentary);
        setAnnotations(result.annotations || []);
        
        setDebugInfo({
          hasFrame: !!result.frame,
          frameSize: result.frame?.length || 0,
          messageCount: 2,
          lastUpdate: new Date().toISOString()
        });
        
        lastAnalysisTimeRef.current = currentTime;
      }
    } catch (error) {
      console.error('Analysis error:', error);
    }

    if (isAnalyzingRef.current) {
      analysisLoopRef.current = requestAnimationFrame(analyzeWebcamFrame);
    } else {
      clearAnalysis();
    }
  };

  const toggleRecording = () => {
    if (!isRecording) {
      clearAnalysis(); // Clear before starting new recording
      setIsRecording(true);
      isAnalyzingRef.current = true;
      lastAnalysisTimeRef.current = 0;
      analyzeWebcamFrame();
    } else {
      setIsRecording(false);
      clearAnalysis(); // Clear when stopping recording
    }
  };

  useEffect(() => {
    clearAnalysis();
    setIsRecording(false);
  }, [mode, clearAnalysis]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const renderAnnotationLabel = (annotation: PersonAnnotation) => {
    if (annotation.class) {
      return annotation.class;
    }
    
    const details = [];
    if (annotation.gender) details.push(annotation.gender);
    if (annotation.ageRange) details.push(annotation.ageRange);
    if (annotation.expression) details.push(annotation.expression);
    
    return details.join(' • ');
  };

  const switchToCamera = useCallback(() => {
    if (videoSource) {
      URL.revokeObjectURL(videoSource);
    }
    setVideoSource(null);
    setVideoFile(null);
    setMode('camera');
    clearAnalysis();
  }, [videoSource, clearAnalysis]);

  if (!isModelLoaded) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-[#1a1b1e] to-[#2b2d31] text-white flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-500" />
          <p className="text-xl font-medium">Loading AI Model...</p>
          <p className="text-sm text-gray-400 mt-2">This may take a moment</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#f8f9fa] to-[#e9ecef] dark:from-[#1a1b1e] dark:to-[#2b2d31] text-gray-900 dark:text-white flex flex-col">
      {/* Sidebar Menu */}
      <div 
        className={`fixed inset-y-0 left-0 w-full md:w-80 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm transform transition-transform duration-300 ease-in-out z-50 ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } shadow-2xl border-r border-gray-200 dark:border-gray-800`}
      >
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-gray-200 dark:border-gray-800">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <Video className="w-6 h-6 text-blue-600 dark:text-blue-500" />
                <h2 className="text-xl font-semibold">Video Analysis AI</h2>
              </div>
              <button
                onClick={() => setIsSidebarOpen(false)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition"
              >
                <X size={20} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto">
            <div className="p-4">
              <div className="space-y-4">
                {/* Camera Switch Button - Only show when video is uploaded */}
                {mode === 'upload' && (
                  <div className="mb-4">
                    <button
                      onClick={() => {
                        switchToCamera();
                        setIsSidebarOpen(false);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 rounded-lg hover:bg-blue-100 dark:hover:bg-blue-500/20 transition text-left"
                    >
                      <Camera size={18} />
                      <span>Switch to Live Camera</span>
                    </button>
                  </div>
                )}

                {/* Upload Section */}
                <div className="mb-8">
                  <h3 className="text-lg font-medium mb-4">Upload Video</h3>
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition ${
                      isDragActive 
                        ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-500/10' 
                        : 'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600'
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
                  <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800/50 transition text-left">
                    <Settings size={18} className="text-gray-500" />
                    <span>Settings</span>
                  </button>
                  <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800/50 transition text-left">
                    <Terminal size={18} className="text-gray-500" />
                    <span>Debug Console</span>
                  </button>
                  <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800/50 transition text-left">
                    <Info size={18} className="text-gray-500" />
                    <span>About</span>
                  </button>
                  <button className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800/50 transition text-left">
                    <Github size={18} className="text-gray-500" />
                    <span>GitHub</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Theme Toggle in Sidebar Footer */}
          <div className="border-t border-gray-200 dark:border-gray-800 p-4">
            <button
              onClick={toggleTheme}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800/50 transition text-left"
            >
              {theme === 'dark' ? <Sun size={18} className="text-gray-500" /> : <Moon size={18} className="text-gray-500" />}
              <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Fixed Header */}
      <header className="fixed top-0 left-0 right-0 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm z-40 border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-4 h-16 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <Video className="w-6 h-6 text-blue-600 dark:text-blue-500" />
            <h1 className="text-xl font-semibold">Video Analysis AI</h1>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowDebug(!showDebug)}
              className={`p-2 rounded-lg transition ${
                showDebug 
                  ? 'bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400' 
                  : 'hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
              title="Toggle Debug Info"
            >
              <Bug size={20} />
            </button>
            <button
              onClick={toggleTheme}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition"
            >
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition"
            >
              <Menu size={20} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 mt-16 px-4 py-6">
        <div className="max-w-7xl mx-auto">
          {error && (
            <div className="mb-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-600 dark:text-red-400 text-sm">
              <div className="flex items-center gap-2">
                <X size={16} className="flex-shrink-0" />
                <p>{error}</p>
              </div>
            </div>
          )}

          {/* Debug Panel */}
          {showDebug && (
            <div className="mb-4 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
              <h3 className="text-sm font-semibold mb-2 text-blue-600 dark:text-blue-400">Debug Information</h3>
              <div className="space-y-1 text-xs text-blue-600 dark:text-blue-400">
                <p>Frame Captured: {debugInfo.hasFrame ? '✅' : '❌'}</p>
                <p>Frame Size: {(debugInfo.frameSize / 1024).toFixed(2)} KB</p>
                <p>Messages to GPT: {debugInfo.messageCount}</p>
                <p>Last Update: {new Date(debugInfo.lastUpdate).toLocaleTimeString()}</p>
              </div>
            </div>
          )}

          {/* Responsive Layout Container */}
          <div className="grid lg:grid-cols-5 gap-6">
            {/* Video Section */}
            <div className="lg:col-span-3 bg-white dark:bg-gray-900 rounded-2xl shadow-xl overflow-hidden border border-gray-200 dark:border-gray-800">
              <div className="p-4 h-full flex flex-col">
                <div ref={videoContainerRef} className="relative flex-1 rounded-xl overflow-hidden">
                  {mode === 'upload' && videoSource ? (
                    <div className="h-full flex flex-col">
                      <video
                        src={videoSource}
                        controls
                        className="w-full h-full object-contain rounded-xl overflow-hidden"
                      />
                      <button
                        onClick={handleQuestionSubmit}
                        disabled={isLoading}
                        className="w-full mt-4 flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white rounded-xl transition disabled:opacity-50 disabled:cursor-not-allowed"
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
                    <div className="h-full relative rounded-xl overflow-hidden">
                      <div className={`${isMobile ? 'w-full aspect-square object-cover' : ''}`}>
                        <Webcam
                          ref={webcamRef}
                          audio={false}
                          className="w-full h-full object-fit"
                          screenshotFormat="image/jpeg"
                          videoConstraints={{
                            facingMode,
                            aspectRatio: 16/9,
                            width: { ideal: 1920 },
                            height: { ideal: 1080 }
                          }}
                        />
                      </div>
                      <div className="absolute bottom-4 right-4 flex gap-2">
                        <button
                          onClick={toggleCamera}
                          className="p-3 rounded-full bg-gray-900/80 hover:bg-gray-800 text-white transition backdrop-blur-sm"
                          title="Switch Camera"
                        >
                          <FlipHorizontal size={20} />
                        </button>
                        <button
                          onClick={toggleRecording}
                          className={`p-3 rounded-full transition backdrop-blur-sm ${
                            isRecording 
                              ? 'bg-red-600 hover:bg-red-700' 
                              : 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600'
                          } text-white`}
                        >
                          {isRecording ? <Pause size={20} /> : <Play size={20} />}
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
                          backgroundColor: 'rgba(59, 130, 246, 0.1)',
                          borderRadius: '0.5rem'
                        }}
                      >
                        <div className="absolute -top-8 left-0 bg-blue-600 dark:bg-blue-500 text-white px-2 py-1 rounded-lg text-xs whitespace-nowrap">
                          {renderAnnotationLabel(annotation)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Analysis Section */}
            <div className={`lg:col-span-2 ${isMobile ? 'h-[40vh]' : 'h-[calc(100vh-8rem)]'}`}>
              <div className="bg-white dark:bg-gray-900 p-6 rounded-2xl shadow-xl h-full border border-gray-200 dark:border-gray-800 flex flex-col">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 flex-shrink-0">
                  <MessageSquare size={20} className="text-blue-600 dark:text-blue-500" />
                  Analysis Results
                </h3>
                {analysis ? (
                  <div className="flex-1 min-h-0">
                    <div className="h-full overflow-y-auto scrollbar-thin">
                      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-xl">
                        <p className="whitespace-pre-wrap text-sm leading-relaxed">{analysis}</p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-gray-400 dark:text-gray-600">
                    <p className="text-sm">Start recording or upload a video to see analysis</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
