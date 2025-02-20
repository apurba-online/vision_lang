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

export async function generateDetailedDescription(sceneData: {
  people: Array<{
    pose: string;
    position: string;
    activity: string;
    movement: string;
  }>;
  objects: string[];
}): Promise<string> {
  if (!openai) {
    return generateBasicDescription(sceneData);
  }

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: `You are a scene analyzer describing people in real-time. Write natural, flowing paragraphs (3-4 sentences) that include:
            - Person's apparent gender and age range
            - Their clothing (colors and style)
            - What they're doing (movements, activities)
            - Any phone/device usage
            - Notable facial expressions
            
            Example: "A young woman in her mid-20s wearing a blue blazer and white blouse is standing in the center of the frame. She's actively engaged with her phone while occasionally glancing up, suggesting she might be on a video call. Her business casual attire and upright posture indicate she's in a professional setting."
            
            Keep descriptions concise but informative. Focus on what's clearly visible.`
        },
        {
          role: "user",
          content: `Describe this scene naturally:
            People: ${JSON.stringify(sceneData.people, null, 2)}
            Objects: ${sceneData.objects.join(', ')}`
        }
      ],
      max_tokens: 250,
      temperature: 0.7
    });

    return response.choices[0]?.message?.content || "Unable to analyze scene.";
  } catch (error) {
    console.error('Error generating detailed description:', error);
    return generateBasicDescription(sceneData);
  }
}

function generateBasicDescription(sceneData: {
  people: Array<{
    pose: string;
    position: string;
    activity: string;
    movement: string;
  }>;
  objects: string[];
}): string {
  if (sceneData.people.length === 0) {
    return "No people detected in the scene.";
  }

  const descriptions = sceneData.people.map((person, index) => {
    const parts = [];
    
    // Create a more natural sentence structure
    parts.push(`A person is ${person.pose} ${person.position}`);
    
    if (person.movement !== 'staying still') {
      parts.push(`they are ${person.movement}`);
    }
    
    if (person.activity) {
      parts.push(`while ${person.activity}`);
    }
    
    // Join with appropriate punctuation
    return parts.join(', ') + '.';
  });

  let description = descriptions.join(' ');
  
  if (sceneData.objects.length > 0) {
    description += ` Nearby objects include ${sceneData.objects.join(' and ')}.`;
  }

  return description;
}