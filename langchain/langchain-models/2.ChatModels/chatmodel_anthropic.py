from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model = 'claude-3.5-sonnet-2024-1022')
result = model.invoke('Footballer`s with most goals')

print(result.content)