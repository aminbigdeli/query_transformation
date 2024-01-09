# Learning to Jointly Transform and Rank Difficult Queries
This repository contains the code and resources for our proposed approach to learn to rank documents and also transform query representations in tandem such that the representation of queries are transformed into one that shows higher resemblance to their relevant document. The main focus of this approach is to integrate two forms of triplet loss functions into neural rankers such that they ensure that each query is moved along the embedding space, through the transformation of its embedding representation, in order to be placed close to its relevant document(s).

## Resources
We have released all the run files, models trained on the original training dataset of MS MARCO, and our proposed datasets that consists of query reformulation triplets to help the community reproduce our results. Due to the file size limitations on github, fine-tuned models and run files are uploaded [here](https://drive.google.com/drive/folders/1-_tp9MnQngkGOTz5-OErGMjnBJ_Xe1wa?usp=sharing).

## Retrieval Effectiveness over all Queries
The main purpose of our proposed approach is to help difficult queries achieve better retrieval effectiveness. As such, and while the focus of is on difficult queries, we report the performance of our proposed approach compared to the performance of the various state-of-the-art neural baselines over MS MARCO dev small queries. As shown in Table 1, the performance of our proposed approach is competitive to that of the baseline methods over all the dev small queries.
<table>
<thead>
  <tr>
    <th>Architecture</th>
    <th>LLM</th>
    <th>Original</th>
    <th>Ours</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">Sentence Transformer</td>
    <td>BERT<br>(base)</td>
    <td>0.334 <a href="https://drive.google.com/file/d/113kYWQesfk753SoyNIYzKU72lgrDW3FH/view?usp=sharing"> (Run)</td>
    <td>0.337 <a href="https://drive.google.com/file/d/1g3E5QYonDHBAfiRZf9VzEfWFMGJgZW4j/view?usp=sharing"> (Run)</td>
  </tr>
  <tr>
    <td>MiniLM<br>(base)</td>
    <td>0.319 <a href="https://drive.google.com/file/d/1jMrs20y5CHfBKFlFYXwk10Exg0PJas9y/view?usp=sharing"> (Run)</td>
    <td>0.325&dagger; <a href="https://drive.google.com/file/d/1k8phzQRr65Dbea7YUK6ZBBkHagDbDvAi/view?usp=sharing"> (Run)</td>
  </tr>
  <tr>
    <td>DistilRoBERTa<br>(base)</td>
    <td>0.305 <a href="https://drive.google.com/file/d/1wv9sUJS0fCLuEc7sf4fLFC4J61fHniA8/view?usp=sharing"> (Run)</td>
    <td>0.308 <a href="https://drive.google.com/file/d/14xTBmCPZl6AQqB2RguhaLHmSfqfJqPTb/view?usp=sharing"> (Run)</td>
  </tr>
  <tr>
    <td>ColBERT</td>
    <td>BERT<br>(base)</td>
    <td>0.338 <a href="https://drive.google.com/file/d/1f1Lfk-LAWHu59QhN0mIFbQCiltR0HbQI/view?usp=sharing"> (Run)</td>
    <td>0.342&dagger; <a href="https://drive.google.com/file/d/1Sp0N89HfKIH0SnCnnZdAm2Q3ovd_SBIi/view?usp=sharing"> (Run)</td>
  </tr>
  <tr>
    <td>RepBERT</td>
    <td>BERT<br>(base)</td>
    <td>0.287 <a href="https://drive.google.com/file/d/1ukJX2NH18DfFnHUAcbhXM4kiH0xe-h-l/view?usp=sharing"> (Run)</td>
    <td>0.290&dagger; <a href="https://drive.google.com/file/d/1DfdvRrVUYkL1BieVjw00W_vUcifxYetk/view?usp=sharing"> (Run)</td>
  </tr>
</tbody>
</table>

## Retrieval Effectiveness over Difficult Queries
To investigate the impact of our porposed approach over difficult queries, we consider difficult queries for a ranking method to be those that have a poor
performance when retrieved by that ranking method. To identify such queries, we rank-order queries based on their MRR@10 values and choose the bottom 50% of queries to represent difficult queries. Given there are 6,980 queries in the small dev set, the bottom 50% of the queries include 3,490 queries in total. In Table 2, the performance (MRR@10) of the different neural ranking models trained based on the original dataset
and our proposed one on difficult queries are compared.
<table>
<thead>
  <tr>
    <th>Architecture</th>
    <th>LLM</th>
    <th>Training</th>
    <th>0-10%</th>
    <th>10-20%</th>
    <th>20-30%</th>
    <th>30-40%</th>
    <th>40-50%</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="6">Sentence Transformer</td>
    <td rowspan="2">BERT<br>(base)</td>
    <td>Original</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0.002</td>
    <td>0.133</td>
  </tr>
  <tr>
    <td>Our Approach</td>
    <td>0.024&dagger;</td>
    <td>0.02&dagger;</td>
    <td>0.022&dagger;</td>
    <td>0.026&dagger;</td>
    <td>0.148&dagger;</td>
  </tr>
  <tr>
    <td rowspan="2">MiniLM<br>(base)</td>
    <td>Original</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0.096</td>
  </tr>
  <tr>
    <td>Our Approach</td>
    <td>0.022&dagger;</td>
    <td>0.017&dagger;</td>
    <td>0.015&dagger;</td>
    <td>0.023&dagger;</td>
    <td>0.115&dagger;</td>
  </tr>
  <tr>
    <td rowspan="2">DistilRoBERTa<br>(base)</td>
    <td>Original</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0.062</td>
  </tr>
  <tr>
    <td>Our Approach</td>
    <td>0.021&dagger;</td>
    <td>0.023&dagger;</td>
    <td>0.022&dagger;</td>
    <td>0.019&dagger;</td>
    <td>0.078&dagger;</td>
  </tr>
  <tr>
    <td rowspan="2">ColBERT</td>
    <td rowspan="2">BERT<br>(base)</td>
    <td>Original</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0.017</td>
    <td>0.142</td>
  </tr>
  <tr>
    <td>Our Approach</td>
    <td>0.015&dagger;</td>
    <td>0.013&dagger;</td>
    <td>0.021&dagger;</td>
    <td>0.034&dagger;</td>
    <td>0.166&dagger;</td>
  </tr>
  <tr>
    <td rowspan="2">RepBERT</td>
    <td rowspan="2">BERT<br>(base)</td>
    <td>Original</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0.04</td>
  </tr>
  <tr>
    <td>Our Approach</td>
    <td>0.011&dagger;</td>
    <td>0.016&dagger;</td>
    <td>0.014&dagger;</td>
    <td>0.018&dagger;</td>
    <td>0.05&dagger;</td>
  </tr>
</tbody>
</table>

## Difference between the cosine similarity of queries and their relevant judged documents over Difficult Queries
To assess whether our query transformation has been able to reduce the distance (increase similarity) between the query and its relevant document, we plot a help-hurt diagram based on the cosine similarity between a query and its relevant document before and after our learnt transformation. The help-hurt diagram is depicted in the Figure below. The diagram shows the difference between the cosine similarity obtained from the embeddings of the transformed query and that of the relevant document and
the cosine similarity of the original query and the relevant document. Positive values show that the transformed queries were closer to the relevant document whereas negative values show the opposite. As seen in the figure, in all rankers, the transformed query is much closer to the relevant document than the original query. This shows that our proposed approach for transforming queries is effectively moving query representations in the right direction.
<p align="center">
  <img src="https://github.com/aminbigdeli/query_transformation/blob/main/embeddings_simlarity.png">
</p>
