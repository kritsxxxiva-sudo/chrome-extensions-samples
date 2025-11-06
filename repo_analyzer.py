#!/usr/bin/env python3
"""
Awesome LLM Apps Repository Comprehensive Analyzer
Expert AI Agent Architect system for understanding and working with the repository
"""

import re
from collections import defaultdict
import json

class AwesomeLLMAppsAnalyzer:
    """Comprehensive analyzer for awesome-llm-apps repository"""
    
    def __init__(self, structure_file):
        self.structure_file = structure_file
        self.projects = []
        self.categories = defaultdict(list)
        self.tech_stack = defaultdict(set)
        
    def parse_structure(self):
        """Parse the directory tree structure"""
        print("📂 Parsing repository structure...")
        
        with open(self.structure_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_path = []
        project_dirs = set()
        
        for line in lines[1:]:  # Skip header
            # Match tree structure: spaces + (├──|└──|│) + name
            match = re.match(r'^(\s*)(├──|└──|│\s+├──|│\s+└──)\s+(.+)$', line)
            if not match:
                continue
            
            indent_str = match.group(1)
            name = match.group(3).strip()
            
            # Calculate depth (each level is 4 spaces)
            depth = len(indent_str.replace('│', ' ')) // 4
            
            # Update path
            if depth < len(current_path):
                current_path = current_path[:depth]
            
            if name.endswith('/'):
                # Directory
                dir_name = name.rstrip('/')
                if depth == len(current_path):
                    if current_path:
                        current_path[-1] = dir_name
                    else:
                        current_path.append(dir_name)
                else:
                    current_path.append(dir_name)
            elif name == 'README.md':
                # Found a project (directory with README)
                if len(current_path) > 1:
                    project_path = '/'.join(current_path)
                    project_name = current_path[-1]
                    
                    self.projects.append({
                        'name': project_name,
                        'path': project_path,
                        'category': self._categorize(project_path)
                    })
        
        print(f"✅ Found {len(self.projects)} projects")
        return self.projects
    
    def _categorize(self, path):
        """Categorize project based on path"""
        path_lower = path.lower()
        
        if 'autonomous_game_playing' in path_lower:
            return 'Autonomous Gaming Agents'
        elif 'agent_teams' in path_lower or 'agent_team' in path_lower:
            return 'Multi-Agent Teams'
        elif 'multi_agent' in path_lower:
            return 'Multi-Agent Systems'
        elif 'mcp_ai_agents' in path_lower or 'mcp_' in path_lower:
            return 'MCP (Model Context Protocol)'
        elif 'rag_tutorials' in path_lower or 'rag_' in path_lower:
            return 'RAG Applications'
        elif 'framework_crash_course' in path_lower:
            return 'Framework Tutorials'
        elif 'advanced_llm_apps' in path_lower:
            return 'Advanced LLM Apps'
        elif 'voice' in path_lower or 'audio' in path_lower:
            return 'Voice & Audio Agents'
        elif 'vision' in path_lower or 'image' in path_lower or 'multimodal' in path_lower:
            return 'Multimodal Agents'
        elif 'fine' in path_lower and 'tun' in path_lower:
            return 'Finetuning Examples'
        else:
            return 'Other'
    
    def categorize_all(self):
        """Group all projects by category"""
        print("🏷️  Categorizing projects...")
        
        for project in self.projects:
            category = project['category']
            self.categories[category].append(project)
        
        print(f"✅ Created {len(self.categories)} categories")
        return self.categories
    
    def generate_comprehensive_docs(self):
        """Generate comprehensive documentation"""
        print("📝 Generating documentation...")
        
        md = self._generate_header()
        md += self._generate_overview()
        md += self._generate_category_catalog()
        md += self._generate_architecture_patterns()
        md += self._generate_comparison_tables()
        md += self._generate_use_case_examples()
        md += self._generate_tech_stack()
        md += self._generate_quick_start()
        md += self._generate_optimization_tips()
        
        return md
    
    def _generate_header(self):
        return """# 🤖 Awesome LLM Apps - Expert AI Agent Architect Guide

> 🔍 **Analyzing:** shubhamsaboo/awesome-llm-apps repository  
> **Role:** Expert AI Agent Architect  
> **Expertise:** Deep knowledge of agent patterns, frameworks, and architectures

---

## 📋 Navigation

1. [Repository Overview](#-repository-overview)
2. [Complete Project Catalog](#-complete-project-catalog)
3. [Architecture Patterns](#-architecture-patterns)
4. [Framework Comparisons](#-framework-comparisons)
5. [Use Case Examples](#-use-case-examples)
6. [Tech Stack Reference](#-tech-stack-reference)
7. [Quick Start Guides](#-quick-start-guides)
8. [Optimization Patterns](#-optimization-patterns)

---

"""
    
    def _generate_overview(self):
        total = len(self.projects)
        categories = len(self.categories)
        
        return f"""## 🎯 Repository Overview

The **awesome-llm-apps** repository is a production-grade collection of LLM-powered applications showcasing cutting-edge AI agent architectures.

### 📊 Repository Statistics
- **Total Projects**: {total}
- **Categories**: {categories}
- **Primary Language**: Python (with TypeScript frontends)
- **Frameworks**: PhiData, CrewAI, LangChain, AutoGen, AgentOps

### 🗂️ Main Directory Structure

```
awesome-llm-apps/
├── advanced_ai_agents/          # Complex multi-agent systems
│   ├── autonomous_game_playing/  # Game-playing AI agents
│   └── multi_agent_apps/        # Collaborative agent teams
├── advanced_llm_apps/           # Production LLM applications
├── ai_agent_framework_crash_course/  # Educational tutorials
├── mcp_ai_agents/              # Model Context Protocol agents
└── rag_tutorials/              # RAG implementation examples
```

---

"""
    
    def _generate_category_catalog(self):
        md = """## 📂 Complete Project Catalog

"""
        
        for category in sorted(self.categories.keys()):
            projects = self.categories[category]
            md += f"\n### {category}\n"
            md += f"**Count**: {len(projects)} projects\n\n"
            
            for project in sorted(projects, key=lambda x: x['name']):
                md += f"#### `{project['name']}`\n"
                md += f"- **Path**: `{project['path']}`\n"
                md += f"- **Category**: {category}\n\n"
        
        md += "\n---\n\n"
        return md
    
    def _generate_architecture_patterns(self):
        return """## 🏗️ Architecture Patterns

### Pattern 1: Single Agent with Tools

**Concept**: Autonomous agent with tool-calling capabilities

```python
from phi.agent import Agent
from phi.tools import DuckDuckGo, PythonTools

agent = Agent(
    model="gpt-4",
    tools=[DuckDuckGo(), PythonTools()],
    instructions="You are a research assistant",
    show_tool_calls=True,
    markdown=True
)

response = agent.run("Research quantum computing trends")
```

**Architecture**:
```
User Query → Agent (LLM) → Tool Selection → Tool Execution → Response Generation
                ↑______________|
```

**Use Cases**:
- Research agents
- Data analysis
- Code generation
- Web search tasks

---

### Pattern 2: Multi-Agent Team (Sequential)

**Concept**: Specialized agents working in sequence

```python
from phi.agent import Agent, Team
from phi.model.openai import OpenAIChat

# Define specialized agents
researcher = Agent(
    name="Researcher",
    role="Research and gather information",
    tools=[DuckDuckGo()],
    model=OpenAIChat(id="gpt-4")
)

analyst = Agent(
    name="Analyst",
    role="Analyze research findings",
    model=OpenAIChat(id="gpt-4")
)

writer = Agent(
    name="Writer",
    role="Write comprehensive report",
    model=OpenAIChat(id="gpt-4")
)

# Create team
team = Team(
    agents=[researcher, analyst, writer],
    workflow="sequential",
    show_progress=True
)

result = team.run("Analyze AI market trends for 2025")
```

**Architecture**:
```
User → Researcher → Analyst → Writer → Final Report
       (Gather)    (Process)  (Present)
```

**Use Cases**:
- Content creation pipelines
- Research & analysis workflows
- Report generation
- Due diligence processes

---

### Pattern 3: Multi-Agent Team (Parallel)

**Concept**: Multiple agents working simultaneously

```python
from concurrent.futures import ThreadPoolExecutor

# Parallel execution team
parallel_team = Team(
    agents=[news_agent, social_agent, market_agent],
    workflow="parallel",
    executor=ThreadPoolExecutor(max_workers=3)
)

# All agents run concurrently
results = parallel_team.run("Get latest AI news")
```

**Architecture**:
```
User Query
    ├─→ News Agent ────┐
    ├─→ Social Agent ──├─→ Aggregator → Final Result
    └─→ Market Agent ──┘
```

**Use Cases**:
- Real-time data aggregation
- Parallel research
- Market intelligence
- Multi-source verification

---

### Pattern 4: RAG (Retrieval Augmented Generation)

**Concept**: Knowledge-grounded generation with vector search

```python
from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector

# Setup knowledge base
knowledge_base = PDFKnowledgeBase(
    path="docs/",
    vector_db=PgVector(
        table_name="pdf_documents",
        db_url="postgresql://localhost/vectordb"
    )
)

# Create RAG agent
rag_agent = Agent(
    knowledge_base=knowledge_base,
    search_knowledge=True,
    model=OpenAIChat(id="gpt-4")
)

response = rag_agent.run("What are the key findings in the research papers?")
```

**Architecture**:
```
User Query → Embedding → Vector Search → Retrieved Docs
                                            ↓
LLM Generation ←─────────────────── Context Injection
     ↓
Response
```

**Use Cases**:
- Document Q&A
- Knowledge management
- Customer support
- Legal document analysis

---

### Pattern 5: MCP (Model Context Protocol)

**Concept**: Standardized tool integration protocol

```python
from mcp import MCPServer, Tool

# Define MCP tools
filesystem_tool = Tool(
    name="filesystem",
    description="Read/write files",
    function=filesystem_handler
)

database_tool = Tool(
    name="database",
    description="Query database",
    function=database_handler
)

# Create MCP server
mcp_server = MCPServer(
    tools=[filesystem_tool, database_tool]
)

# Agent with MCP integration
agent = Agent(
    mcp_servers=[mcp_server],
    model="gpt-4"
)
```

**Architecture**:
```
Agent → MCP Protocol → Tool Registry → Tool Execution
         (Standard)                    (Filesystem, DB, API)
```

**Use Cases**:
- Enterprise integration
- Tool standardization
- Cross-platform agents
- Plugin ecosystems

---

### Pattern 6: Autonomous Gaming Agent

**Concept**: Self-playing game agents with decision-making

```python
from phi.agent import Agent
from game_tools import ChessEngine, MoveValidator

chess_agent = Agent(
    name="Chess Master",
    model="gpt-4",
    tools=[ChessEngine(), MoveValidator()],
    instructions='''
    You are a chess master. Analyze board state,
    calculate best moves, and play strategically.
    '''
)

# Autonomous play loop
while not game.is_over():
    move = chess_agent.run(f"Current board: {game.board()}")
    game.make_move(move)
```

**Architecture**:
```
Game State → Agent Analysis → Move Decision → Game Update
     ↑_______________|
```

**Use Cases**:
- Game AI
- Strategic decision-making
- Simulation testing
- Reinforcement learning

---

"""
    
    def _generate_comparison_tables(self):
        return """## 📊 Framework Comparisons

### Agent Frameworks Comparison

| Framework | Best For | Complexity | Multi-Agent | RAG Support | Tool Calling |
|-----------|----------|------------|-------------|-------------|--------------|
| **PhiData** | Production apps | Low-Medium | ✅ Teams | ✅ Built-in | ✅ Extensive |
| **CrewAI** | Role-based teams | Medium | ✅ Crews | ⚠️ External | ✅ Custom |
| **LangChain** | RAG & chains | Medium-High | ⚠️ Complex | ✅ Excellent | ✅ Good |
| **AutoGen** | Conversations | Medium | ✅ Groups | ⚠️ Limited | ⚠️ Basic |
| **AgentOps** | Monitoring | Low | ❌ | ❌ | ❌ |

### Model Selection Guide

| Model | Best For | Cost | Speed | Context | Reasoning |
|-------|----------|------|-------|---------|-----------|
| **GPT-4 Turbo** | Production | $$$ | Fast | 128K | Excellent |
| **GPT-4o** | Multimodal | $$$ | Fast | 128K | Excellent |
| **o1/o3** | Deep reasoning | $$$$ | Slow | 128K | Best |
| **Claude 3 Opus** | Long context | $$$$ | Medium | 200K | Excellent |
| **Claude 3 Sonnet** | Balanced | $$ | Fast | 200K | Very Good |
| **Llama 3 70B** | Open source | $ | Fast | 8K | Good |

### Vector Database Comparison

| Database | Setup | Scale | Performance | Cost |
|----------|-------|-------|-------------|------|
| **Pinecone** | Easy | Excellent | Fast | $$$ |
| **ChromaDB** | Very Easy | Good | Medium | Free |
| **FAISS** | Medium | Excellent | Very Fast | Free |
| **Qdrant** | Easy | Very Good | Fast | $$ |
| **PgVector** | Medium | Good | Good | $ |

---

"""
    
    def _generate_use_case_examples(self):
        return """## 💡 Use Case Examples

### Example 1: Financial Analysis Agent Team

**Scenario**: Analyze stock market and generate investment report

**Architecture**:
```
User: "Analyze AAPL stock"
  ↓
Market Researcher (DuckDuckGo, YFinance)
  ↓
Financial Analyst (Pandas, Calculations)
  ↓
Risk Assessor (Monte Carlo, VaR)
  ↓
Report Writer (Markdown generation)
  ↓
Final Investment Report
```

**Code Pattern**:
```python
# agents/finance_team.py
market_researcher = Agent(
    name="Market Researcher",
    tools=[YFinanceTools(), DuckDuckGo()],
    instructions="Gather market data and news"
)

analyst = Agent(
    name="Financial Analyst",
    tools=[PythonTools()],
    instructions="Perform financial analysis"
)

risk_assessor = Agent(
    name="Risk Assessor",
    instructions="Evaluate investment risks"
)

writer = Agent(
    name="Report Writer",
    instructions="Create comprehensive investment report"
)

finance_team = Team(
    agents=[market_researcher, analyst, risk_assessor, writer],
    workflow="sequential"
)
```

---

### Example 2: RAG Document Q&A System

**Scenario**: Answer questions from company documentation

**Architecture**:
```
PDF Documents → Chunking → Embeddings → Vector DB
                                           ↓
User Question → Query Embedding → Similarity Search
                                           ↓
Retrieved Chunks → Context → LLM → Answer
```

**Code Pattern**:
```python
# rag/document_qa.py
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.chroma import ChromaDB

kb = PDFKnowledgeBase(
    path="company_docs/",
    vector_db=ChromaDB(
        collection="company_docs",
        path="./chroma_db"
    )
)

# Load and index documents
kb.load()

qa_agent = Agent(
    knowledge_base=kb,
    search_knowledge=True,
    instructions="Answer questions based on company documentation"
)

answer = qa_agent.run("What is our return policy?")
```

---

### Example 3: Travel Planning Multi-Agent System

**Scenario**: Plan complete trip with flights, hotels, itinerary

**Architecture**:
```
User Requirements
  ├─→ Destination Agent (research locations)
  ├─→ Flight Agent (search flights)
  ├─→ Hotel Agent (find accommodations)
  ├─→ Food Agent (restaurant recommendations)
  ├─→ Itinerary Agent (daily schedule)
  └─→ Budget Agent (cost analysis)
       ↓
Complete Travel Plan
```

**Code Pattern**:
```python
# teams/travel_planner.py
destination_agent = Agent(
    name="Destination Expert",
    tools=[DuckDuckGo()],
    instructions="Research destinations and attractions"
)

flight_agent = Agent(
    name="Flight Finder",
    tools=[KayakFlightTool(), GoogleFlightsTool()],
    instructions="Find best flight options"
)

hotel_agent = Agent(
    name="Hotel Specialist",
    tools=[KayakHotelTool()],
    instructions="Find suitable accommodations"
)

itinerary_agent = Agent(
    name="Itinerary Planner",
    instructions="Create day-by-day itinerary"
)

travel_team = Team(
    agents=[destination_agent, flight_agent, hotel_agent, itinerary_agent],
    workflow="sequential"
)
```

---

"""
    
    def _generate_tech_stack(self):
        return """## 🔧 Tech Stack Reference

### Core Frameworks

#### PhiData
```python
# Installation
pip install phidata

# Quick start
from phi.agent import Agent
agent = Agent(model="gpt-4")
agent.run("Hello")
```

**Strengths**:
- Production-ready
- Built-in observability
- Easy team coordination
- Excellent documentation

---

#### CrewAI
```python
# Installation
pip install crewai

# Quick start
from crewai import Agent, Task, Crew

agent = Agent(role="Researcher", goal="Find information")
task = Task(description="Research AI trends", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
crew.kickoff()
```

**Strengths**:
- Role-based design
- Task management
- Process workflows
- Good for complex teams

---

#### LangChain
```python
# Installation
pip install langchain

# Quick start
from langchain.agents import create_openai_functions_agent
agent = create_openai_functions_agent(llm, tools, prompt)
```

**Strengths**:
- Extensive integrations
- RAG excellence
- Chain composition
- Large ecosystem

---

### Vector Databases

#### ChromaDB (Local)
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(documents=texts, ids=ids)
results = collection.query(query_texts=["search"])
```

#### Pinecone (Cloud)
```python
import pinecone

pinecone.init(api_key="your-key")
index = pinecone.Index("docs")
index.upsert(vectors=embeddings)
results = index.query(vector=query_embedding, top_k=5)
```

---

### Tools & Integrations

| Category | Tools |
|----------|-------|
| **Search** | DuckDuckGo, Tavily, Google Search, Serper |
| **Web Scraping** | BeautifulSoup, Playwright, Selenium, Scrapy |
| **Data Processing** | Pandas, Polars, NumPy |
| **Embeddings** | OpenAI, Cohere, Sentence Transformers |
| **Monitoring** | AgentOps, LangSmith, Phoenix |
| **APIs** | REST, GraphQL, gRPC |

---

"""
    
    def _generate_quick_start(self):
        return """## 🚀 Quick Start Guides

### Guide 1: Your First Agent

```python
# 1. Install
pip install phidata openai

# 2. Create agent.py
from phi.agent import Agent
from phi.model.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4"),
    instructions="You are a helpful assistant"
)

# 3. Run
agent.print_response("What is AI?")
```

---

### Guide 2: Agent with Tools

```python
# 1. Install with tools
pip install phidata duckduckgo-search

# 2. Create research_agent.py
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo

agent = Agent(
    tools=[DuckDuckGo()],
    show_tool_calls=True
)

# 3. Run research
agent.print_response("Latest AI news")
```

---

### Guide 3: Multi-Agent Team

```python
# 1. Install
pip install phidata openai

# 2. Create team.py
from phi.agent import Agent, Team

researcher = Agent(name="Researcher", role="Research topics")
writer = Agent(name="Writer", role="Write articles")

team = Team(agents=[researcher, writer])

# 3. Run team
team.print_response("Write article about AI")
```

---

### Guide 4: RAG System

```python
# 1. Install
pip install phidata chromadb pypdf

# 2. Create rag_agent.py
from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.chroma import ChromaDB

kb = PDFKnowledgeBase(
    path="docs/",
    vector_db=ChromaDB(collection="docs")
)
kb.load()

agent = Agent(knowledge_base=kb, search_knowledge=True)

# 3. Ask questions
agent.print_response("Summarize the documents")
```

---

"""
    
    def _generate_optimization_tips(self):
        return """## ⚡ Optimization Patterns

### 1. Agent Memory Management

**Problem**: Agent forgets context in long conversations

**Solution**: Implement memory systems

```python
from phi.memory import Memory
from phi.memory.db.postgres import PostgresMemory

agent = Agent(
    memory=PostgresMemory(
        db_url="postgresql://localhost/agent_memory",
        table_name="conversations"
    ),
    add_history_to_messages=True,
    num_history_responses=5  # Keep last 5 exchanges
)
```

**Optimization**:
- Use semantic memory for long-term storage
- Implement forgetting mechanisms
- Compress old conversations

---

### 2. Tool Call Optimization

**Problem**: Too many unnecessary tool calls

**Solution**: Implement caching and selective calling

```python
from functools import lru_cache
from phi.tools import tool

@tool
@lru_cache(maxsize=100)
def expensive_api_call(query):
    # Cached tool that calls expensive API
    # Results cached for repeated queries
    return "api_result"

agent = Agent(
    tools=[expensive_api_call],
    tool_choice="auto",  # Let LLM decide when to call
)
```

**Best Practices**:
- Cache API responses
- Batch similar requests
- Use tool descriptions wisely
- Set timeout limits

---

### 3. Multi-Agent Coordination

**Problem**: Agents duplicate work or conflict

**Solution**: Implement proper workflow patterns

```python
# Sequential for dependent tasks
sequential_team = Team(
    agents=[data_collector, analyzer, reporter],
    workflow="sequential"  # Each waits for previous
)

# Parallel for independent tasks
parallel_team = Team(
    agents=[news_agent, market_agent, social_agent],
    workflow="parallel"  # All run simultaneously
)

# Custom coordination
def coordinator(task):
    # Decide which agents to use
    if task.requires_research:
        return [researcher, analyst]
    return [writer]

smart_team = Team(
    agents=[researcher, analyst, writer],
    coordinator=coordinator
)
```

---

### 4. RAG Performance Tuning

**Problem**: Slow retrieval or irrelevant results

**Solution**: Optimize chunking and search

```python
from phi.knowledge.pdf import PDFKnowledgeBase

kb = PDFKnowledgeBase(
    path="docs/",
    # Optimize chunking
    chunk_size=512,      # Smaller = more precise
    chunk_overlap=50,    # Overlap for context
    
    # Optimize search
    vector_db=ChromaDB(
        collection="optimized_docs"
    ),
    
    # Improve results
    num_documents=3,     # Return top 3
    rerank=True          # Re-rank results
)
```

**Best Practices**:
- Experiment with chunk sizes
- Use hybrid search (keyword + semantic)
- Implement re-ranking
- Monitor retrieval quality

---

### 5. Cost Optimization

**Problem**: High API costs

**Solution**: Strategic model selection

```python
# Use cheaper models for simple tasks
cheap_agent = Agent(model="gpt-3.5-turbo")

# Use expensive models only when needed
smart_router = Agent(
    model="gpt-3.5-turbo",
    instructions='''
    For complex reasoning, use gpt-4.
    For simple tasks, handle yourself.
    '''
)

# Token management
agent = Agent(
    max_tokens=500,           # Limit output
    temperature=0.1,          # Reduce randomness
    streaming=True,           # Stream for UX
    response_model=MyModel    # Structured output
)
```

**Cost Saving Tips**:
- Cache responses
- Use prompt compression
- Implement rate limiting
- Monitor token usage

---

### 6. Error Handling & Resilience

**Problem**: Agent crashes on errors

**Solution**: Implement robust error handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def reliable_agent_call(agent, query):
    try:
        return agent.run(query)
    except Exception as e:
        print(f"Error: {e}, retrying...")
        raise

# Use in agent
agent = Agent(
    retry_strategy=retry(stop=stop_after_attempt(3)),
    timeout=30,  # 30 second timeout
    fallback_model="gpt-3.5-turbo"  # Fallback if primary fails
)
```

---

## 🎓 Learning Path

### Beginner
1. Start with single agent + basic tools
2. Experiment with different models
3. Add memory and conversation history
4. Build a simple RAG system

### Intermediate
5. Create multi-agent teams
6. Implement custom tools
7. Add monitoring & logging
8. Optimize performance

### Advanced
9. Build autonomous agents
10. Implement MCP protocol
11. Create agent frameworks
12. Deploy to production

---

## 📚 Additional Resources

### Official Documentation
- [PhiData Docs](https://docs.phidata.com)
- [CrewAI Docs](https://docs.crewai.com)
- [LangChain Docs](https://docs.langchain.com)

### Community
- [PhiData Community](https://community.phidata.com)
- [CrewAI Discord](https://discord.gg/crewai)
- [LangChain Discord](https://discord.gg/langchain)

---

**Generated by Awesome LLM Apps Analyzer**  
*Expert AI Agent Architect System*
"""

def main():
    analyzer = AwesomeLLMAppsAnalyzer('awesome-llm-apps-structure.txt')
    
    # Parse and categorize
    analyzer.parse_structure()
    analyzer.categorize_all()
    
    # Generate docs
    docs = analyzer.generate_comprehensive_docs()
    
    # Save documentation
    with open('AWESOME_LLM_APPS_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(docs)
    
    # Generate JSON catalog
    catalog = {
        'total_projects': len(analyzer.projects),
        'categories': {cat: len(projs) for cat, projs in analyzer.categories.items()},
        'projects': analyzer.projects
    }
    
    with open('project_catalog.json', 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)
    print(f"\n📊 Statistics:")
    print(f"   Total Projects: {len(analyzer.projects)}")
    print(f"   Categories: {len(analyzer.categories)}")
    print(f"\n📝 Files Generated:")
    print(f"   ✓ AWESOME_LLM_APPS_GUIDE.md (Complete guide)")
    print(f"   ✓ project_catalog.json (JSON catalog)")
    print(f"\n🎯 Categories:")
    for cat, projs in sorted(analyzer.categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   {cat}: {len(projs)} projects")

if __name__ == '__main__':
    main()
