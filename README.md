# offensive-word-exploration

__INTRO:__
Often at times, bad words as a sole criterion are not that great for identifying cyberbullying instances. Posts from an anonymous QA site show that bad words can also be associated with highly sexual posts, but these sexual posts can be encouraged by the user who the post is targeted to. This project utilized two datasets with one containing instances of cyberbullying and the other consisting of highly sexual posts, but are not considered cyberbullying. For privacy reasons I cannot upload the dataset I collected, but can upload the script used for exploration as well as the visualization produced. 

I located the top 30 bad words in the cyberbullying dataset and likewise with the sexual posts dataset. From these 60 words, set operations were utilized to seperate words that occure in cyberbullying cases, highly sexual (but not cyberbullying) conversations, and words that occure in both cases. Leveraging PCA and t-SNE we can compare these two algorithms while at the same time explore these bad words. Word2vec (skip-gram) was used for producing the word embeddings.




![simple graph comparing t-SNE and PCA](https://github.com/Kosmos01/offensive-word-exploration/blob/master/t-SNE.PCA.SG.png)


NOTE: Words found in the visualization are highly inappropriate
