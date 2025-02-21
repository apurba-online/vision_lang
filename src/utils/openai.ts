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
    
    // If we're near the rate limit, wait until the next reset
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

    // Add a small delay between requests
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
        
        // Create a detailed scene description including age and expression
        const sceneContext = sceneData.people.map(person => {
          const age = person.annotation?.ageRange || 'unknown age';
          const expression = person.annotation?.expression || 'neutral expression';
          return `A person (${age}, ${expression}) is ${person.pose} ${person.position}${person.movement !== 'staying still' ? `, ${person.movement}` : ''}${person.activity ? `, ${person.activity}` : ''}.`;
        }).join(' ');

        const objectContext = sceneData.objects.length > 0
          ? `Nearby objects include ${sceneData.objects.join(' and ')}.`
          : '';

        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content: "You are a natural scene descriptor. Describe the image in 3-4 concise sentences, incorporating the provided age and expression details. Focus on creating a natural, flowing description that includes physical appearance, expressions, and environmental details. Avoid technical terms."
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
                  text: `Describe this scene naturally, incorporating these details: ${sceneContext} ${objectContext}`
                }
              ]
            }
          ],
          max_tokens: 150,
          temperature: 0.7
        });
        
        const description = response.choices[0]?.message?.content || "Unable to analyze scene.";
        RATE_LIMIT.currentTokens += Math.ceil(description.length / 3);
        resolve(description);
      } catch (error) {
        console.error('GPT-4 API error:', error);
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
}): string {
  if (sceneData.error) {
    return sceneData.error;
  }

  if (sceneData.people.length === 0 && sceneData.objects.length === 0) {
    return "No objects or people detected in the scene.";
  }

  const descriptions = sceneData.people.map(person => {
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
    return parts.join(', ') + '.';
  });

  let description = descriptions.join(' ');
  
  if (sceneData.objects.length > 0) {
    description += ` Nearby objects include ${sceneData.objects.join(' and ')}.`;
  }

  return description;
}
