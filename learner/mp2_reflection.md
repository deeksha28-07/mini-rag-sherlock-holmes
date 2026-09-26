# MP2 Reflection

## What worked

My RAG system worked well and got 5/5 correct source matches. The chunks and citations helped answer from the correct Sherlock Holmes story.

## What didn't work

For question 1, facts matched was only 1/5 even though the answer and source were correct. One Blue Carbuncle question also retrieved one extra unrelated story.

## What I'd change

I would test different chunk sizes and k values to improve retrieval. I would also add reranking to remove unrelated chunks.

## One surprise

I was surprised that text-embedding-3-small found the correct source for all five questions. I learned that correct source match is more important than exact wording in fact matching.