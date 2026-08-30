<div align="center">

# Reviewer2

### Revisor Técnico de Conteúdo com IA

*Um segundo revisor independente para as explicações técnicas dos seus vídeos.*

**[🇬🇧 Read in English](README.md)**

</div>

---

## Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Sentence_Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white)
![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![pytest](https://img.shields.io/badge/pytest_·_196_testes-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

</div>

| Camada | Tecnologia | Papel |
|---|---|---|
| 🎬 **Áudio/vídeo** | FFmpeg | extração e normalização da faixa de áudio |
| 🎙️ **Transcrição** | faster-whisper | ASR local com timestamps e confiança por segmento |
| 🧠 **LLM** | Ollama · llama.cpp · qualquer API compatível com OpenAI | extração de afirmações e análise crítica |
| 🔤 **Embeddings** | Sentence Transformers (multilíngue) | busca semântica, inclusive entre idiomas |
| 🗂️ **Banco vetorial** | FAISS (NumPy como alternativa) | recuperação de evidências |
| 📄 **Documentos** | PyMuPDF · pypdf · python-docx · BeautifulSoup | PDF, DOCX, HTML, Markdown, TXT, URLs |
| ✅ **Validação** | Pydantic | contratos de dados de ponta a ponta |
| 🌐 **Interface web** | FastAPI · Uvicorn | envio dos arquivos e acompanhamento pelo navegador |
| 🧪 **Testes** | pytest | 196 testes, offline, ~20 s |

Tudo roda **localmente e de graça**. Nenhuma API paga é necessária e nenhum arquivo sai da
sua máquina.

---

## O que é o Reviewer2

Você grava uma aula. Ela está tecnicamente correta? Onde exatamente você simplificou demais,
generalizou sem base, ou afirmou algo que a sua própria bibliografia contradiz?

O Reviewer2 responde isso. Você entrega **um vídeo** e **um ou mais materiais de
referência**; ele transcreve a fala, extrai as afirmações técnicas que você fez, busca
evidências nos seus materiais e audita a explicação afirmação por afirmação.

O relatório abre com um veredito acionável:

| Veredito | O que significa |
|---|---|
| ✅ `PODE_PUBLICAR` | nenhum erro técnico sustentado por evidência |
| 🟡 `AJUSTES_MENORES` | sem erros graves; vale reformular ou anotar na descrição |
| 🟠 `PRECISA_CORRECAO` | há afirmações que podem gerar compreensão incorreta |
| 🔴 `REGRAVAR_TRECHO` | há erro que compromete a explicação |

### O que ele identifica

| | |
|---|---|
| erros técnicos | afirmações parcialmente corretas |
| imprecisões de formulação | contradições internas do próprio vídeo |
| contradições com as fontes | omissões relevantes |
| generalizações indevidas | causalidade não demonstrada |
| definições incorretas | simplificações pedagógicas |

### Por que confiar no que ele aponta

**Citação inventada é impossível por construção.** O modelo nunca escreve uma citação: ele
aponta o identificador de uma frase que recebeu, e o texto é buscado na fonte. Um
identificador inexistente é descartado antes de chegar ao relatório.

**Toda crítica passa por um advogado do diabo.** Uma segunda etapa ataca cada achado com
sete perguntas — existe leitura alternativa? a fonte realmente sustenta isso? a transcrição
pode ter criado o problema? Uma crítica que a fonte não sustenta é removida.

**A confiança é calibrada, não copiada do modelo.** O valor exibido combina a confiança
declarada com a qualidade da transcrição, os scores de recuperação e a concordância entre
as fontes. Um veredito sem citação verificada nunca aparece com confiança alta.

**Incerteza é um resultado válido.** Quando suas fontes não cobrem um ponto, ele diz isso
explicitamente — e essa contagem fica **separada** dos erros, porque é um limite do seu
material de referência, não um erro seu.

### Medido, não prometido

Execução real sobre o exemplo do repositório com `qwen2.5:7b-instruct`, comparada a um
gabarito humano pelo `reviewer2-eval`:

| Métrica | Resultado |
|---|---|
| Precisão das críticas | **83,3%** |
| Taxa de falsos positivos | **0%** |
| Cobertura de evidência | **100%** |
| Recall na extração de afirmações | **85,7%** |
| Qualidade da recuperação | **100%** |

> Um único vídeo de exemplo não é um benchmark. Meça no *seu* material com o
> `reviewer2-eval` — o projeto traz a ferramenta justamente para isso.

---

## Como instalar

### 1. Requisitos

* Python 3.10+
* [FFmpeg](https://ffmpeg.org/) no `PATH`
* [Ollama](https://ollama.com/) (ou llama.cpp) para o modelo de linguagem

### 2. Instalar o projeto

```bash
git clone <seu-repositório> reviewer2
cd reviewer2

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Em máquina **sem GPU**, instale o PyTorch em versão CPU **antes** — evita baixar ~3 GB de
bibliotecas CUDA inúteis:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Depois:

```bash
pip install -r requirements.txt    # tudo
pip install -e .                   # disponibiliza os comandos reviewer2*
```

Instalação mínima (só o núcleo; suficiente para Ollama e para o modo `--offline`):

```bash
pip install -r requirements-minimal.txt
```

### 3. Instalar o modelo

```bash
# instalação do Ollama: https://ollama.com/download
ollama serve                       # normalmente já sobe como serviço
ollama pull qwen2.5:7b-instruct    # ~4,7 GB, uma única vez
```

| Modelo | Download | RAM durante a revisão | Qualidade |
|---|---|---|---|
| `qwen2.5:3b-instruct` | ~2 GB | ~4 GB | básica |
| `qwen2.5:7b-instruct` | ~4,7 GB | ~7 GB | **recomendado** |
| `qwen2.5:14b-instruct` | ~9 GB | ~12 GB | melhor |

O modelo só ocupa RAM **durante** a revisão. O Ollama o descarrega após 5 minutos ocioso;
`ollama stop qwen2.5:7b-instruct` libera na hora.

### 4. Verificar a instalação

```bash
reviewer2 --check
```

Confere FFmpeg, dependências, servidor do modelo, RAM e disco, e diz exatamente o que fazer
em cada item que faltar.

```
  ✓ Python                             3.13.15
  ✓ FFmpeg                             /usr/bin/ffmpeg
  ✓ faster-whisper                     transcrição de áudio
  ✓ LLM (ollama/qwen2.5:7b-instruct)   respondendo em http://localhost:11434
  ✓ Memória                            24 GB no total · o modelo precisa de ~7 GB

  Tudo pronto para revisar.
```

Os modelos de Whisper e de embeddings são baixados sozinhos no primeiro uso.

---

## Como usar

### Linha de comando

```bash
reviewer2 --video aula.mp4 --reference artigo.pdf
```

Gera três arquivos:

```text
data/reports/aula_review.md             # o relatório
data/reports/aula_review_audit.json     # cada crítica com sua cadeia de evidências
data/reports/aula_review_retrieval.json # cada consulta, trecho e score de similaridade
```

### Mais exemplos

```bash
# várias referências: arquivos, um diretório inteiro e uma URL
reviewer2 --video aula.mp4 \
          --reference paper.pdf \
          --reference notas.md \
          --reference bibliografia/ \
          --reference https://pt.wikipedia.org/wiki/Memória_cache

# reaproveitar uma transcrição já feita (pula a etapa de ASR)
reviewer2 --transcript data/transcripts/aula.json --reference paper.pdf

# relatório em inglês
reviewer2 --video aula.mp4 --reference paper.pdf --report-language en

# execução seca, sem nenhum modelo — para ver o formato de saída em segundos
reviewer2 --transcript examples/aula_transcricao.json \
          --reference examples/referencia_arquitetura.md --offline
```

### Interface web

```bash
pip install 'reviewer2[api]'
reviewer2-web                      # http://127.0.0.1:8000
```

Envie o vídeo e as referências pelo navegador, acompanhe o progresso das seis etapas e leia
ou baixe o relatório na própria página. Escuta apenas em `localhost` por padrão; use
`--host 0.0.0.0 --port 8080` para expor na rede.

### Medir a qualidade da revisão

```bash
reviewer2-eval --report data/reports/aula_review_audit.json \
               --gold examples/gold_standard.json
```

Anote você mesmo o gabarito de um vídeo (formato em `examples/gold_standard.json`) e veja
a precisão real do modelo escolhido. É assim que se decide se vale trocar de modelo — com
número, não com impressão.

### Anatomia do relatório

Cada afirmação vira uma cadeia completa: **crítica → afirmação → timestamp → evidência →
página da fonte → justificativa**.

```markdown
### Claim #003

**Timestamp:** 00:00:26

**Afirmação:**
> "A memória cache é uma memória não volátil que fica dentro do processador."

**Classificação:** `INCORRETA`   **Gravidade:** `CRITICO`   **Confiança:** `ALTA` (0.78)

**Evidência:**
Fonte: `arquitetura.pdf` · Página: 12 · Similaridade: 0.81
> "A memória cache é uma memória volátil construída com células SRAM."

**Análise:** a fonte afirma explicitamente o oposto: a cache é volátil.

**Correção sugerida:**
> "A memória cache é uma memória volátil que fica dentro do processador."
```

### Opções úteis

| Opção | Significado |
|---|---|
| `--check` | verifica o ambiente e sai |
| `--offline` | sem LLM e sem download: motor heurístico (execução seca) |
| `--llm-model` | troca o modelo, ex.: `qwen2.5:14b-instruct` |
| `--llm-provider` | `ollama` (padrão), `llamacpp`, `openai-compatible`, `heuristic` |
| `--llm-url` | endereço do servidor do modelo |
| `--whisper-model` | `tiny` … `large-v3` |
| `--top-k`, `--min-score` | amplitude da recuperação e limiar de evidência |
| `--no-verification` | pula o advogado do diabo (mais rápido, menos preciso) |
| `--no-omissions`, `--no-contradictions` | pula essas análises |
| `--force-transcription`, `--rebuild-index` | ignora os caches |
| `--report-language` | `pt` ou `en` |
| `--print-config` | mostra a configuração efetiva e sai |

---

## Configurações

A configuração fica fora do código, em `config.yaml` e `.env` (copie o `.env.example`).

**Precedência:** CLI › variáveis de ambiente › `config.yaml` › padrões.

```bash
export REVIEWER2_LLM__MODEL=llama3.1:8b       # prefixo REVIEWER2_, __ para aninhamento
export REVIEWER2_RETRIEVAL__TOP_K=8
```

### Principais chaves

```yaml
llm:
  provider: ollama          # ollama | llamacpp | openai-compatible | heuristic
  model: qwen2.5:7b-instruct
  base_url: http://localhost:11434
  temperature: 0.1          # baixo: isto é auditoria, não escrita criativa
  num_ctx: 8192             # contexto; principal fator de RAM depois do modelo
  max_tokens: 3072
  concurrency: 2            # chamadas simultâneas (ver Desempenho)
  fallback_to_heuristic: false   # false = falha explícita, nunca uma revisão falsa

retrieval:
  top_k: 6                  # trechos recuperados por afirmação
  min_score: 0.25           # abaixo disso a afirmação é NAO_SUSTENTADA
  backend: auto             # auto | faiss | numpy

analysis:
  cache_critiques: true     # reaproveita críticas entre execuções
  max_contradiction_pairs: 10    # cada par comparado é uma chamada ao modelo
  detect_omissions: true

verification:
  enabled: true             # o advogado do diabo
  drop_unsustained: true    # remove crítica que a fonte não sustenta

transcription:
  model: small              # tiny | base | small | medium | large-v3
  language: null            # null = detecta sozinho
```

### Trocar o modelo de LLM

Todo o sistema conversa apenas com a interface `LLMProvider`, então o backend é trocado por
configuração, sem tocar no código de análise.

```yaml
# Ollama (padrão)
llm: { provider: ollama, model: qwen2.5:7b-instruct, base_url: "http://localhost:11434" }

# llama.cpp, LM Studio, vLLM — qualquer servidor compatível com a API OpenAI
llm: { provider: llamacpp, model: local-model, base_url: "http://localhost:8080" }

# Endpoint na nuvem (não usa a RAM da sua máquina; seus arquivos saem dela)
llm: { provider: openai-compatible, model: llama-3.1-8b-instant,
       base_url: "https://api.groq.com/openai", api_key: null }
```

Para adicionar um provedor novo, implemente `generate()` e registre em `build_llm()`:

```python
class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, *, system=None, temperature=None,
                 max_tokens=None, stop=None) -> str: ...
```

### Desempenho

O tempo é dominado pela geração de tokens do modelo — em CPU, **~47 s por chamada** com o
`qwen2.5:7b-instruct` (6,3 tok/s), sendo ~85% disso geração e apenas ~15% leitura do prompt.
Três ajustes já vêm ligados:

| Ajuste | Efeito medido |
|---|---|
| `llm.concurrency: 2` | 2 chamadas simultâneas rendem **2,7×** mais que 1; com 3 a vazão cai |
| `llm.cache: true` | **medido: 2060 s → 28 s** ao repetir uma revisão; execução interrompida retoma de onde parou |
| `max_contradiction_pairs: 10` | cada par comparado é uma chamada |

Um vídeo de 10 minutos (~40 afirmações) gera cerca de **77 chamadas**, o que dá **50 a 60
minutos** na primeira execução. Reprocessar depois custa segundos.

**Não troque por um modelo menor para ganhar tempo.** Medição comparativa neste projeto:

| | `qwen2.5:7b-instruct` | `qwen2.5:3b-instruct` |
|---|---|---|
| Latência por chamada | 47,3 s | 26,2 s (1,8× mais rápido) |
| **Problemas encontrados** (vídeo com 5 erros reais) | **6** | **0** |
| Cobertura de evidência | 100% | 0% |
| Erro de calibração | 0,220 | 0,533 |

O 3B extrai e recupera bem, mas não sustenta o raciocínio do advogado do diabo: descarta
6 de 7 críticas e devolve um relatório vazio. Metade do tempo por nenhum erro encontrado
não é economia.

---

## Arquitetura

### O caminho dos dados

```text
        VÍDEO                          REFERÊNCIAS (PDF · DOCX · MD · HTML · URL)
          │                                       │
          ▼                                       ▼
   ffmpeg: extrai o áudio            parsing (página e seção preservadas)
          │                                       │
          ▼                                       ▼
   faster-whisper                          chunking com sobreposição
   timestamps + confiança                        │
          │                                       ▼
          ▼                                Sentence Transformers
   janelas semânticas                             │
          │                                       ▼
          ▼                                FAISS · banco vetorial
   extração de afirmações (LLM) ──────────────────┤
          │                                       │
          ▼                                       ▼
   por afirmação: retrieval ────────────► evidências + scores + procedência
          │
          ▼
   crítica presa à evidência (LLM)
          │   ├── citação por identificador de frase
          │   ├── omissões na mesma resposta
          │   └── contradições internas do vídeo
          ▼
   advogado do diabo — segunda camada adversarial
          │
          ▼
   calibração de confiança
          │
          ▼
   relatório Markdown + JSON de auditoria + rastro de recuperação
```

### Estrutura do código

```text
reviewer2/
├── app/
│   ├── transcription/   ffmpeg + faster-whisper
│   │   ├── audio.py         extração e normalização do áudio
│   │   ├── whisper.py       ASR, timestamps, confiança por segmento
│   │   └── models.py        serialização das transcrições
│   ├── documents/       ingestão das referências
│   │   ├── loader.py        arquivos, diretórios e URLs
│   │   ├── parser.py        PDF, TXT, Markdown, DOCX, HTML
│   │   └── chunker.py       chunking com página e seção preservadas
│   ├── embeddings/      Sentence Transformers + fallback por hashing
│   ├── retrieval/       FAISS / NumPy + recuperador com rastro auditável
│   ├── llm/             interface abstrata + Ollama, llama.cpp, heurístico
│   ├── analysis/        o núcleo da auditoria
│   │   ├── claims.py          transcrição → afirmações analisáveis
│   │   ├── critique.py        RAG + julgamento preso à evidência
│   │   ├── contradictions.py  contradições internas do vídeo
│   │   ├── omissions.py       condições da fonte ausentes no vídeo
│   │   ├── confidence.py      calibração da confiança
│   │   └── cache.py           reaproveitamento entre execuções
│   ├── verification/    advogado do diabo
│   ├── metrics/         suíte de avaliação (precisão, falsos positivos, calibração)
│   ├── reports/         gerador de Markdown (pt/en)
│   ├── web/             interface FastAPI (fila de jobs + página única)
│   ├── prompts/         todos os prompts, com a política anti-alucinação
│   ├── doctor.py        verificação do ambiente (--check)
│   ├── pipeline.py      orquestração das etapas
│   └── main.py          CLI
├── data/                vídeos, documentos, transcrições, relatórios (fora do git)
├── examples/            transcrição, referência e gabarito de exemplo
└── tests/               196 testes: unitários + integração ponta a ponta
```

### Camadas substituíveis

Cada camada fica atrás de uma interface, então o backend é escolhido por configuração:

| Camada | Interface | Implementações |
|---|---|---|
| LLM | `LLMProvider` | Ollama · llama.cpp / OpenAI-compatible · heurístico offline |
| Embeddings | `EmbeddingModel` | Sentence Transformers · hashing determinístico |
| Banco vetorial | `VectorStore` | FAISS · NumPy (força bruta exata) |

Se uma dependência opcional faltar, o sistema degrada para a alternativa disponível e
**registra isso no relatório**, em vez de falhar em silêncio.

### Rastreabilidade

Cada execução grava três arquivos: o relatório, um `*_audit.json` com todas as críticas e
suas cadeias de evidência, e um `*_retrieval.json` com todas as consultas ao banco vetorial,
os trechos recuperados e seus scores. Qualquer conclusão pode ser auditada até a consulta
que a produziu.

### Testes

```bash
pytest                     # 196 testes, offline, ~20 s
pytest --cov=app           # com cobertura
```

Cobrem transcrição, parsing, chunking, embeddings, retrieval, extração, classificação,
contradições, calibração, verificação, relatório, cache, provedores HTTP, métricas e
interface web — além de um teste de integração que verifica que **nenhuma citação do
relatório existe fora do corpus indexado**.

---

## Licença

Distribuído sob a **Licença MIT**. Veja [LICENSE](LICENSE) para o texto completo.

```
Copyright (c) 2026 Gustavo Rech Costa

Concede-se permissão, livre de encargos, a qualquer pessoa que obtenha uma cópia
deste software e dos arquivos de documentação associados, para negociar no
software sem restrição, incluindo usar, copiar, modificar, mesclar, publicar,
distribuir, sublicenciar e/ou vender cópias do software.
```
