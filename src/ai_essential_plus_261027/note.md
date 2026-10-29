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

# Day-2

- SLM/sLLM
  - 10B 정도를 기준으로 LLM와 sLLM 을 분류함 
  - 요즘 27B, 35B 모델들이 인기가 있음 
  - Task Length (작업 시간)을 작업의 난이도록 보고 있음. 현재 위치는 4시간 정도 걸리는 작업임

- Word2Vec
  - https://projector.tensorflow.org/ 200 차원 Vector를 'PCA'라는 것을 이용해서 3차원으로 표시해 놓은 것임    

- 최적화 구조
  - 신경망 구조
    - Dense vs MoE 
      - Gemma 4 Dense / MOE 
      - Qwen Dense / MoE
        - MoE 26B - 활성화 3.8B
      - Dense 
        - 성능 / 정확도 우위 
        - Dense 구조를 CPU로 Offload 시키면, 1 TPS 로 내려올 수 있다. (사람이 읽는 속도 20 TPS 정도)
      - MoE
        - 속도 / 지연 시간 / 메모리
        - MoE를 CPU로 Offload 할 수 있다. Active Expert는 GPU에서, In-Active Expert는 CPU에서 
        - 여러 명을 위한 서비스를 위해서는 MoE 가 사용성 측면에서 나을 수 있음

        https://hfviewer.com/google/gemma-4-26B-A4B-it
        https://hfviewer.com/google/gemma-4-26B-A4B
      
  - 멀티 모달 입력
   https://bbycroft.net/llm
   - Multimodal SLM
     - 하단의 MLP - Text Vector와 Image Vector의 연계를 맺어 주는 역할 (Connector)
       - 차원 수도 맞춰 놓고, Vector 공간도 정리해 두었음 
     - 입력 구성
       - Concatenate 
         - Text Vector 와 Image Vector를 그냥 이어 붙임 (Qwen3 VL)
           - Context Length가 길어짐
       - Token FUsion
         - 아예 Vector를 Text와 Image를 섞어 버림 (Gemma 모델)
   - LLaVA: 최신 모델도 기본적으로는 LLaVA 뼈대에서 유래함 (학습 비용이 낮음) 
   - Native Multimoal
   - Vision SLMs Training 
     - Instruction Tuning 
     - 멀티 모달 데이터 학습 
       - MLP Projector (LLaVA)
       - Q-Former (관심에서 많이 멀어져 있음)
       - Image Tokenize (Native Multimodal)

  - 학습 
    - 학습 데이터의 구조 
      - Pre Training - Text Completion 학습
      - Instruction Tuning 
    - 학습 데이터 
      - LAION-5B 데이터 셋
        - Query / Image / Caption  
      - OCR (Optical Character Recognition)

  - 추론
    - Quantization 
      - 양자화 --> 경량화 방법 
      - LLM 가중치 표현을 어떻게 잘 할 것인가 ?  
      - bfloat16(BF16) : float32를 16bit float 으로 바꿔도 성능이 잘 유지됨. (사실상 표준)
        - Google Brain에서 만든 것이라서 'b' 가 붙었다. (Exponent는 유지하고 Fraction을 줄여도 영향이 크게 없더라..)  
      - QLoRA (Quantized Low-Rank Adaptation)
        - PEFT   
      - 모델 가중치 포맷 
        - Safetensors (원본)
        - Safetensors --> GGUF (양자화/llama.cpp 용) 변환 
        - AWQ (vLLM 용)
      - 양자화 할 떄 성능 저하를 측정하는 지표
        - Top 1% Accuracy   
        - KLD - KL Divergence     

- start_llama_server 


# Day-3 
- Fine Tuning
  - effective batch size = gradient accumulatoin steps x per device train batch size 
  - epoch = 1 // fine tuning은 한자리 수로 하는 것이 정립되어 있는 것이 정립되어 있음
    - fine tuning data로 과적합 되거나, pretraining 되는 것의 지식을 잃어 버리는 것을 방지하기 위함 
  - learning rate는 요새는 거의 고정한다. training FW에서 제공하는 learning rate를 그대로 써도 된다.

- Vision Understanding & Generation
  - CNN --> ViT 로 파라다임 시프트가 일어 있다. 
  - Feature 추출이 중요하다. (Automatic Feature Extraction)
  - ViT 
    - Patch 단위로 자르고 Vector로 바꾼다.
    - Vector에서 축은 특징을 의미한다. 

  - VAE
    - 속도 안정성 균형 
  - Flow
    - 속도 느림, 안정성 좋음 
  - GAN
    -속도 빠름, 안정성 떨어짐 (Training 힘듬)
  - Diffusion
    - https://huggingface.co/docs/diffusers/v0.37.0/ko/using-diffusers/img2img
    - Ghibili Diffusion

  - Optical Character Recognition (OCR)
    - Docling 
      - Table Former
      - Doc Lay Net
      - Rapid OCR (기본 엔진)
    - Document Understanding  
    - E2E 
      - VLM에 통쨰로 (디버깅이 좀 어렵다)

# 미니 프로젝트





 