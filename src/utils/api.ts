export async function analyzeVideo(videoFile: File, question: string) {
  try {
    // Convert video file to base64
    const videoBase64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result as string;
        // Remove data URL prefix
        resolve(base64.split(',')[1]);
      };
      reader.onerror = reject;
      reader.readAsDataURL(videoFile);
    });

    const response = await fetch(import.meta.env.VITE_HUGGINGFACE_MODEL_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${import.meta.env.VITE_HUGGINGFACE_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: {
          video: videoBase64,
          question: question
        }
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API request failed: ${error}`);
    }

    const result = await response.json();
    return {
      answer: typeof result === 'string' ? result : result.generated_text || 'No analysis available'
    };
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
}