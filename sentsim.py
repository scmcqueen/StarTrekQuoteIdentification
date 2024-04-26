#  Note: All of this code is also in train_sent_sim.ipynb, with explanation

from sentence_transformers import SentenceTransformer, models, losses, evaluation
import pandas as pd
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
data = pd.read_csv('sbertdata.csv')
data.drop('Unnamed: 0', axis=1, inplace=True)

train_df, other_df = train_test_split(
    data, test_size=.3, random_state=13, stratify=data['similarity'])

test_df, dev_df = train_test_split(
    other_df, test_size=.333, random_state=13, stratify=other_df['similarity'])

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
dev_df.dropna(inplace=True)

word_embedding_model = models.Transformer(
    "bert-base-uncased", max_seq_length=256)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model], device='cuda')

train_examples = []
for index, row in train_df.iterrows():
    ex = InputExample(
        texts=[row['quote_x'], row['quote_y']], label=float(row['similarity']))
    train_examples.append(ex)


train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

evaluator = evaluation.EmbeddingSimilarityEvaluator(list(dev_df['quote_x']), list(
    dev_df['quote_y']), [float(x) for x in list(dev_df['similarity'])])

model.fit(train_objectives=[
          (train_dataloader, train_loss)], epochs=5, warmup_steps=100, evaluator=evaluator, save_best_model=True, checkpoint_save_total_limit=5, evaluation_steps=50)

model.save('sentsimmodel_eval')
