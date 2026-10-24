# DADA2 — Agentic RAG over SEC filings

**Fill in both sections at the bottom before you submit.** Keep the deployed
app line exactly in this form; it is how your app gets found and graded.

## Getting started

There is no separate spec. The instructions are in the notebooks, and
`DADA2_00_Start_Here.ipynb` is the one to open first: it explains the
assignment, the setup, the grading, and what each of the four notebooks after
it asks for. Work through them in order.

Get the files the same way as every class day: pull, then work in a copy:

```bash
cd lecture-files
git pull
cd ..
cp -r lecture-files/DADA2 DADA2     # work in the copy, not in lecture-files/
cd DADA2
cp example.env .env                  # then fill in your keys
```

Then open `DADA2_00_Start_Here.ipynb` from your `DADA2` copy and run its
setup check.

Deploying is `DEPLOY.md`.

---

## Submission

```
Deployed app: <your Render URL>
```

ADD YOUR NAME TO YOUR RENDER PROJECT SO THE LINK CAN BE PERSONALIZED.

PLEASE ENSURE YOUR URL WORKS BY TESTING IT IN INCOGNITO MODE.

### Write-up

Paste the write-up from notebook 04 (TODO 4.3) here, under the same five
headings.

**Conditions.** I used `gemini-3.5-flash-lite` as the chat model. The same model is also the judge in `evaluate.judge_answer`. The decision model is `clef-flash` on Cloudflare and the reranker is Cohere `rerank-v3.5`. For embeddings I used `qwen3-embedding-0.6b` on Cloudflare, which gives 1024 dimensions. The 27 sections were split into 2,124 chunks of 1,200 characters with an overlap of 150. The agents retrieve k = 5 with a rerank pool of 10 and `RRF_K` = 60. The thresholds are `SURE` = 0.6, `ABSENT_SURE` = 0.8 and `MAX_ATTEMPTS` = 3. I scored the routers on all 24 labelled questions and the agents on `specific[:8] + comparative[:6]`. Every table is from a single run. Cell 12 is a second run of only the diagnosis agent, which I did to find out where it went wrong.

**The routers.** The decision model got 20 out of 24 correct and the chat model got 19 out of 24. The decision model took 0.27 s per call compared to 1.02 s for the chat model, so it is around 3.8 times faster. A difference of one question is too small to say that one router is more accurate than the other. The real benefit of the decision model is something else. It gives a probability for every option and its calls are not counted in the Gemini quota. `SURE` makes use of this probability. When the router is not confident enough, the agent falls back to `filter+rrf`.

**The agents.** In cell 9 the ladder scored 0.93 (13 of 14) with 4.9 chat calls and 15.6 s per question. The diagnosis agent scored 0.79 (11 of 14) with 2.7 chat calls, 2.3 decision calls and 10.7 s per question. The biggest difference was on the question about NVIDIA's target markets for fiscal 2024 (cell 12). The route was correct (specific, 0.90). Saying that the passages were not enough was also correct, because the ladder's grader also rejected the passages from `filter` and `filter+rrf`. The problem was that `diagnose` picked `incomplete` (0.40) instead of `wrong_words`. Because of this, `decide` sent the question to `plan` and `execute`, which search every step using `with_filter`. That is the lowest rung of the ladder. After three attempts the diagnosis agent replied that the passages did not list the markets. The ladder, on the other hand, went up to `filter+rrf+rerank` and found the answer. I would be careful with the accuracy numbers though, because the judge only sees the question and the answer. In cell 10 it marked an answer wrong for "fictional events from fiscal year 2025 and a hypothetical Intel-Nvidia partnership". But `AMD_2025_risk_factors.txt` clearly says that Nvidia announced this partnership in September 2025. Just one mistake like this would remove half of the 2-question gap, so I do not think the gap is final.

**What did not help.** Multi-query fusion did not help. In notebook 02, on the six specific questions, `filter` and `filter+rrf` both got recall@5 of 1.00 and MRR of 0.833 (cells 17 and 27). Still, fusion adds one extra chat call and three extra Pinecone searches for every question. The reason is in `filter_rrf` itself. Every rewrite is searched with the same `where`, so all four lists come from the same company and the same year. The filter alone was already putting a gold file in the top five for every question. So fusion could only change the order of the results. The MRR did not move at all. The real gains came from two other places. The filter took recall from 0.67 to 1.00. The reranker then took MRR from 0.833 to 0.889 (cell 34).

**What I would ship.** I would ship the diagnosis agent. It needs 2.7 chat calls per question in place of 4.9. So with the same daily quota it can answer about 1.8 times more questions. It is also about 5 s faster. But I would make one change. Either `incomplete` should go up the `STRONGER` ladder or `execute` should search with `filter+rrf+rerank`. This fixes the NVIDIA miss. It would not have fixed AMD versus Micron though, since that question reached `filter+rrf+rerank` and still ended on `wrong_words`. Before shipping I would like to know how many user questions are about a single filing and how many are comparisons. The savings come mainly from single-filing questions, as 7 of the 8 specific questions in cell 12 needed only one attempt. I would also give the gold passages to the judge before trusting a gap of only two questions.
