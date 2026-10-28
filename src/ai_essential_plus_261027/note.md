# Day-1
## LLM 
- 모델 사이즈(파라 미터 개수) : 수백억 ~ 수조 y = m * x + b (m,b: 파라미터)
- Context Windows : 토큰 (수십만~수백만), Page (수백 ~수천)
- Data Set : 토큰 수 (조단위)

## LLM + a (Augmented LLMs)
- 제어 불가 (Lack of Control). 필요한 task를 잘 수행하지 못한다.
- 필요한 task를 잘 수행하지 못한다.
  - 지식의 한계 (Knowledge Cut Off) - 최신성/전문성 부족
  - 표현의 한계 (Output Style)
  - 이해의 한계 (Understanding)
  - 기억의 한계 (Context Memory)
- 성능 부족 (underperform)
  - 성능 비교 사이트 : https://artificialanalysis.ai/
                     https://arena.ai/leaderboard/chat/text/english 
                     https://benchmarklist.com/benchmarks/?ability=coding_agents#all-benchmarks
  * SSM(State Space Model) -> Non Transformer Model                     
  - Vanilla Attention Mechanism의 한계
    - Lost in the middle (U-shaped Performance Curve) - U자형 성능을 갖는다.
      - Text에서 처음에 나왔던 내용과 마지막에 나온 내용에 대해서는 성능을 잘 갖지만, 중간의 내용에 대해서는 성능이 잘 나오지 않는다. 
      - 한국어: 명사 + 조사 로 구성됨, 오늘 + 은/부터/까지 -> 토큰
      - Attention Mechanism
  - Hallucination
    - NeurIPS
  - Vertical & Horizontal AI
    - 컨텍스트 및 도구 개선 
      - 모델은 고정하고  - 프롬프트, 더 나은 프롬프트 체인, RAG, Workflow, Multi Agent --> Vertical AI Domain 이라고 부른다.
    - Veritcal AI (Cognitive Architecture)
  - Agent Architecture 
    - Agent: LLM - (toos, memory, planning)  
  - ReAact (Reasoning + Action) Prompt -> Prompt 엔지니어링의 계보에 속한다.
    - Thought + Act + Observation : ReActing Loop라고 한다. (생각하고 행동하고 발견한다.)
    - Lang Chain으로 하면 ReAct로 동작한다. 

    * Prompt Engineering 
      - CoT(Chain of Thought), ToT(Tree of Thought) -> 한계: 되돌아 갈 수 가 없다. (루프라는 개념을 생각해 본다면 되돌아 가야 한다)

    - Prompt Principle
    - Prompt Template
    - Few Shot Prompt : 모델이 참조할 수 있는 예시가 달려 있는 프롬프트 (3-shot: 예시가 3개임)
      - 예시를 추가 하는 비용 (예시가 잘못되면 성능이 안나온다.)
    - Zero Shot Prompt : 예시가 없음
    - Prompt Chaining : 이전 출력이 다음 출력의 입력으로 Chainging - 메모리가 필요하다. + 충분한 Context Length가 필요하다.

  - Context / No Memory 
  - AI User 
    - 사람의 일과 인공 지능의 일을 나눈다.
    - 내가 곧 AI 
 
  - Agent Tool Calling / Funtion Calling - Tool Calling 능력을 가진 LLM - Agent가 얼마나 Tool Calling을 잘하느냐 
  - MCP (Model Context Protocol) 
    - API를 몰라도 API를 쓸 수 있는 계층. API의 상세를 몰라도 된다.
    
  - RAG & Fine-tuning 
    - RAG(Retrieval Augmented Generation): 외부 지식 활용 (DB를 운용한다)
      - Testing Generative Models -> LLM-as-a-Judge 방식으로 평가 - 유사도 Judge
        - Pairwise 평가: A와 B를 제시하고, A에 더 가깝다는 프롬프트를 줌 
      - Adaptive RAG
        - 필요에 따라 RAG 수행 (Agentic RAG)
      - Agent Workflows 
        - 문제를 쪼개서 준다.
      - Multi Agent Workflows 
        - Multi Agent는 각 기능을 담당하는 Agent를 구현한다.
        - Single Point of Failure 
          - Context Length -> 멀티 에이전트로 분담
          - Toll Call -> Tool이 많아 지거나, 호출 못하거나 
          - 작업의 성질이 각각 다른 경우 - 모델 마다 잘하는 것이 다를 수 있음 
            - Multi Agent는 각 기능을 담당하는 Agent를 구현한다.
              - 두뇌는 공통 Wrapper만 여럿 
              - 두뇌도 각각 다르게 
        - Prompt Chaining 
          - LLM call#1 --> Gate --> LLM Call#2 
          - Prompt Routing            

    - Fine Tuning - Instruction Tuning (Behavior를 녹아 들이려면 이게 좋다)

- Skill : 재사용이 가능한 동적 로딩되는 프롬프트 
- Lang-Chain (High Level/추상화가 더 많이 되어 있음), Lang-Graph (Low Level)
  - State 
    - Lang-Chain : 대화 만 있음 
    - Lang-Grpah : 대화 말고도 여러가지를 할 수 있음 
  - 분기/루핑 등 Flow --> Lang Graph 지원
    - Node, Edge (Edge는 항상 Directed Graph 이다)
      

     






 