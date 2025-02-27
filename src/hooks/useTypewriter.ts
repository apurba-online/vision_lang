import { useState, useEffect, useRef } from 'react';

export function useTypewriter(text: string, speed: number = 30, autoScroll: boolean = true) {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const reset = () => {
    setDisplayedText('');
    setIsTyping(false);
  };

  useEffect(() => {
    if (!text) {
      setDisplayedText('');
      setIsTyping(false);
      return;
    }

    setIsTyping(true);
    let currentIndex = 0;
    setDisplayedText(text[0]);
    const interval = setInterval(() => {
      if (currentIndex < text.length) {
        setDisplayedText(prev => {
          const newText = prev + text[currentIndex];
          // Only auto-scroll if enabled
          if (autoScroll && containerRef.current) {
            setTimeout(() => {
              containerRef.current?.scrollTo({
                top: containerRef.current.scrollHeight,
                behavior: 'smooth'
              });
            }, 0);
          }
          return newText;
        });
        currentIndex++;
      } else {
        clearInterval(interval);
        setIsTyping(false);
      }
    }, speed);

    return () => clearInterval(interval);
  }, [text, speed, autoScroll]);

  return { displayedText, isTyping, containerRef, reset };
}