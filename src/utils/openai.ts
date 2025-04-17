import OpenAI from 'openai';

const createOpenAIClient = () => {
  const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
  if (!apiKey) {
    console.warn('OpenAI API key not found. Detailed descriptions will be disabled.');
    return null;
  }
  return new OpenAI({
    apiKey,
    dangerouslyAllowBrowser: true
  });
};

const openai = createOpenAIClient();

const RATE_LIMIT = {
  tokensPerMin: 30000,
  currentTokens: 0,
  lastReset: Date.now(),
  queue: [] as Array<{
    task: () => Promise<void>;
    resolve: (value: string) => void;
    retries: number;
  }>,
  processing: false,
  backoffTime: 1000,
  maxBackoff: 60000,
  maxRetries: 3,
  resetInterval: 60000,
  cooldownThreshold: 0.8,
};

setInterval(() => {
  RATE_LIMIT.currentTokens = 0;
  RATE_LIMIT.lastReset = Date.now();
  RATE_LIMIT.backoffTime = 1000;
}, RATE_LIMIT.resetInterval);

async function processQueue() {
  if (RATE_LIMIT.processing || RATE_LIMIT.queue.length === 0) return;
  
  RATE_LIMIT.processing = true;
  
  while (RATE_LIMIT.queue.length > 0) {
    const queueItem = RATE_LIMIT.queue[0];
    
    if (RATE_LIMIT.currentTokens > RATE_LIMIT.tokensPerMin * RATE_LIMIT.cooldownThreshold) {
      const timeUntilReset = RATE_LIMIT.resetInterval - (Date.now() - RATE_LIMIT.lastReset);
      await new Promise(resolve => setTimeout(resolve, Math.max(timeUntilReset, 1000)));
      continue;
    }

    try {
      await queueItem.task();
      RATE_LIMIT.queue.shift();
      RATE_LIMIT.backoffTime = Math.max(1000, RATE_LIMIT.backoffTime / 2);
    } catch (error) {
      if (error.error?.type === 'tokens' && error.error?.code === 'rate_limit_exceeded') {
        queueItem.retries++;
        
        if (queueItem.retries > RATE_LIMIT.maxRetries) {
          RATE_LIMIT.queue.shift();
          queueItem.resolve('Rate limit exceeded. Please try again in a moment.');
          continue;
        }

        const waitTime = Math.min(
          RATE_LIMIT.maxBackoff,
          RATE_LIMIT.backoffTime * Math.pow(2, queueItem.retries)
        );
        RATE_LIMIT.backoffTime = waitTime;
        
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      }
      
      RATE_LIMIT.queue.shift();
      queueItem.resolve(`Error analyzing scene: ${error.message}`);
    }

    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  RATE_LIMIT.processing = false;
}

function convertDataURLToBase64(dataUrl: string): string {
  const base64Regex = /^data:image\/(png|jpeg|jpg);base64,(.*)$/;
  const matches = dataUrl.match(base64Regex);
  return matches ? matches[2] : dataUrl;
}

export async function generateDetailedDescription(sceneData: {
  people: Array<{
    pose: string;
    position: string;
    activity: string;
    movement: string;
    annotation?: {
      ageRange?: string;
      expression?: string;
    };
  }>;
  objects: string[];
  frame?: string;
  error?: string;
  isScenic?: boolean;
}): Promise<string> {
  if (!openai) {
    return generateBasicDescription(sceneData);
  }

  if (sceneData.error) {
    return `Analysis paused: ${sceneData.error}`;
  }

  if (!sceneData.frame) {
    console.log('No frame data provided');
    return generateBasicDescription(sceneData);
  }

  return new Promise((resolve) => {
    const task = async () => {
      try {
        const base64Image = convertDataURLToBase64(sceneData.frame);
        
        const systemPrompt = sceneData.isScenic
          ? `You are a nature and landscape analyst. Describe the scene in detail, focusing on:
              1. Natural features (waterfalls, mountains, forests, etc.)
              2. Lighting and atmospheric conditions
              3. Colors and textures in the landscape
              4. Environmental details (rock formations, vegetation)
              5. Overall mood and atmosphere of the scene
              6. Time of day and weather conditions if apparent
              7. Scale and perspective of landscape elements
              
              Create a vivid, engaging description that captures the essence of the natural scene. Report it like a journalist. No bullet points or bold words. (Within 100 to 150 words.)`
          : `You are a scene analyst. Describe the scene comprehensively, including:
              1. Overall environment and setting
              2. Notable objects and their arrangement
              3. People and their activities if present
              4. Lighting and atmosphere
              5. Colors and textures
              6. Movement and action
              7. General mood of the scene
              
              If there is anything concerning write that too. Report it like a journalist. No bullet points or bold words. (Within 100 to 150 words.)`;

        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content: systemPrompt
            },
            {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: {
                    url: `data:image/jpeg;base64,${base64Image}`
                  }
                },
                {
                  type: "text",
                  text: "Describe this scene in detail, focusing on its visual elements and atmosphere."
                }
              ]
            }
          ],
          max_tokens: 300,
          temperature: 0.7
        });
        
        const description = response.choices[0]?.message?.content || "Unable to analyze scene.";
        RATE_LIMIT.currentTokens += Math.ceil(description.length / 3);
        resolve(description.trim()); // Keep trim() to remove any trailing whitespace
      } catch (error) {
        console.error('GPT-4o API error:', error);
        resolve(generateBasicDescription(sceneData));
      }
    };

    RATE_LIMIT.queue.push({
      task,
      resolve,
      retries: 0
    });

    processQueue().catch(console.error);
  });
}

function generateBasicDescription(sceneData: {
  people: Array<{
    pose: string;
    position: string;
    activity: string;
    movement: string;
    annotation?: {
      ageRange?: string;
      expression?: string;
    };
  }>;
  objects: string[];
  error?: string;
  isScenic?: boolean;
}): string {
  if (sceneData.error) {
    return sceneData.error;
  }

  if (sceneData.isScenic) {
    return "This video shows a scenic view. The analysis system is focusing on the natural elements and landscape features of the scene.";
  }

  if (sceneData.people.length === 0 && sceneData.objects.length === 0) {
    return "The scene is visible but no specific objects or people are detected. The video appears to show general scenery or environmental elements.";
  }

  const descriptions = [];

  if (sceneData.people.length > 0) {
    sceneData.people.forEach(person => {
      const parts = [];
      const age = person.annotation?.ageRange || 'unknown age';
      const expression = person.annotation?.expression || 'neutral expression';
      
      parts.push(`A person (${age}, ${expression}) is ${person.pose} ${person.position}`);
      if (person.movement !== 'staying still') {
        parts.push(`they are ${person.movement}`);
      }
      if (person.activity) {
        parts.push(`while ${person.activity}`);
      }
      descriptions.push(parts.join(', ') + '.');
    });
  }

  if (sceneData.objects.length > 0) {
    descriptions.push(`The scene contains ${sceneData.objects.join(', ')}.`);
  }

  return descriptions.join(' ').trim(); // Keep trim() to remove any trailing whitespace
}