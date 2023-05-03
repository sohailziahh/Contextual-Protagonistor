
### TL,DR:

> Determine the main character/organization given a piece of text such
> as articles or news.

## Objective

The Contextual Protagonist feature is part of the media engine product. And helps in finding the entity that the text is generally about. The output can than be used to auto-search images of the entity from our database of images in dropbox or from imago.

To identify the main entity we use a coreference model developed by `spacy` and `explosion ai`. Essentially this model takes a piece of text , parses into tokens and resolves the references aka pronouns to their original entity. For example

![The sentence "Philip plays the bass because he loves it.", illustrating that the "he" refers to "Philip" and "it" refers to "the bass"](https://explosion.ai/4cda26959bc6d01ad6274b2e07851d28/coref-1.gif)


Reference: https://explosion.ai/blog/coref

Once the references are resolved, an NER model is applied to then resolved text and the entities present are counted. The entity with the highest count is our main-character or contextual protagonist of the text. We also apply post processing aka heuristics on the output to ensure that edge cases are handled. The following section explains them in more details.

## Problem Formulation

The problem is formalised as statistical, which relies on processed information from CoRef + NER. The reason for this formulation and the reason behind the use of `CoRef` and `NER` will hopefully become a bit clear in a while. We tackled this problem based on our assumption about the nature of content/news/article. The assumption is that, in most cases, when a news/text is about an entity, its emphasised on in the text multiple times. We could have dealt this problem by simply applying NER and extracting all entities of the type PERSON and ORG. But life is not always fair and straightforward. The goal is not to extract all the entities being mentioned in the text, but rather to extract the main entity who/which the article is about. Therefore, we opted to rely on our aforementioned assumption and decided to tackle this problem as statistical. By that we mean, count how many times an extracted entity has been mentioned. And to be able to do that, we need to deal with all the anaphors. If you are not aware of this linguistic terminology, it simply means  `a word or phrase that refers back to an earlier word or phrase, e.g, in Lionel Messi went to the club because he thought he would enjoy. he refers to Lionel Messi. `  This problem of resolving anaphors is known as Coreference Resolution. By definition, _Coreference Resolution is the task of finding all expressions that refer to the same entity in a text_. 


## Methodology

The steps are as follow: `Text` -> `PreProcessing` -> `CoRef` -> `ResolveText` -> `NER` -> `CountEntities` -> `PostProcessing`. 

1) **PreProcessing**: 
Before applying CoRef algorithm directly on the raw input text/news, we first strip the name accents off text as most footballers have accents in their names (e.g, Mandžukić). This may not be necessary but would be nice to have data simple. 

2) **CoRef**:  
We use [Spacy's latest CoRef component](https://spacy.io/api/coref). This is still part of experimental project. Specifically, `spacy-experimental 0.6.0`. Sample code : 
```python
# Commandline
pip install spacy-experimental==0.6.0
pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl

# Python
import spacy
nlp = spacy.load("en_coreference_web_trf")
doc = nlp("The cats were startled by the dog as it growled at them.") 
print(doc.spans)
```
Which will print the coref clusters:
{'coref_clusters_1': [the dog, it], 'coref_clusters_2': [The cats, them]}

3) **Reference Resolution**: After we get the clusters, we then use that to replace the anaphors with the respective antecedents. 

4) **NER**: 
Using Spacy's NER. 

5) **PostProcessing**:
As Coreference Resolution is quite challenging problem and is still an open-research problem, and also as we are using a component from Spacy that is still experimental, predictions/outputs tend to be messy in few occasions. So to make sure that we don't get absolutely bizarre outputs as predicated/extracted protagonists back to the user, we apply some heuristics to overcome this issue to some extent. Some of them are: 

    1) Check the first letter of the predicted protagonist(s).
    The assumption behind this is: The entity(s) about whom the news/article
    is about, always/should have proper naming convention;
    first letter of the word is capitalised.

    2) There are cases where the middle name of the person has all small
    letters. so for that, I only consider the firstand last name. e.g,
    Sohail van Ziahh.

    3) The predicted protagonist cannot have more than 3 words in it.
    (or could be 4; trial and error, we'll see.).
    
    4)  If "'s" is in the token, remove it.


## OpenAI DaVinvi's for Paraphrasing

For paraphrasing tweets, reddit posts, or any text, I also used OpenAI GPT-3 DaVinci. It costs [$0.03 per 1000 tokens](https://openai.com/api/pricing/). Right now, we are using the free tier. Again, we are using Sohail Zia's API key. He has been too generous. Anyways, we have this integrated into the same function. It's output is sent back in the very same response as the protagonistor. 
This is a lot better as its GPT!!
