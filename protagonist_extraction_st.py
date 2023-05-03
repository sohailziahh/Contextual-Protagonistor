import spacy
from spacy.tokens import Doc
from wasabi import msg
from collections import Counter
import streamlit as st
import pandas as pd
import time

# Streamlit version


def resolve_references(doc: Doc) -> str:
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_

            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string


def show_ents(doc):
    entss = list()

    if doc.ents:
        for ent in doc.ents:
            print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
            if ent.label_.__eq__("DATE") or ent.label_.__eq__("ORDINAL") or ent.label_.__eq__("CARDINAL"):
                continue
            entss.append(ent.text)

    else:
        print('No named entities found.')
        return

    predictions = []

    dics = Counter(entss)
    max_value = max(dics, key=dics.get)

    predictions.append(max_value)

    if dics.get(max_value) > 1 and len(dics) > 1:
        lst = sorted(dics.items(), key=lambda item: item[1], reverse=True)
        filtered = get_filtered(lst[1:])
        print(filtered)
        print(check_proper(filtered))
        max_value = max_value + check_proper(filtered)

    htmlstr1 = f"""<p style='background-color:green;
                                               color:white;
                                               font-size:18px;
                                               border-radius:2px;
                                               font-family:serif;                                           
                                               line-height:60px;
                                               padding-left:17px;
                                               opacity:0.8'>
                                               Predcition: Text is about <b>{max_value}</b></style>
                                               <br></p>"""
    st.markdown(htmlstr1, unsafe_allow_html=True)

    print(Counter(entss))


def check_proper(listt):
    propers = ""
    if len(listt) == 0:
        return ""

    for lst in listt:
        flag = False
        for token in lst.split(" "):
            if token[0].isupper():
                flag = True
        if flag:
            propers = propers + ", " + " ".join([lst])

    return propers


def get_filtered(listt):
    filtered = []
    for lst in listt:
        if lst[1] > 1:
            filtered.append(lst[0])

    return filtered


def spacy_ner(text, NER_MODEL):
    doc1 = NER_MODEL(text)
    show_ents(doc1)


def on_button_click():
    st.session_state.counter += 1
    if st.session_state.counter >= len(bunch_of_articles):
        st.session_state.counter = 0

    coref_ner()


def write_title_and_label(st, counter):
    # htmlstr1 = f"""<p style='background-color:blue;
    #                  color:white;
    #                  font-size:18px;
    #                  border-radius:2px;
    #                  font-family:serif;
    #                  line-height:60px;
    #                  padding-left:17px;
    #                  opacity:0.8'>
    #                  <b>Title: {titles[counter]}</b>
    #                  </style>
    #                  <br></p>"""
    # st.markdown(htmlstr1, unsafe_allow_html=True)

    htmlstr1 = f"""<p style='background-color:orange;
                         color:white;
                         font-size:18px;
                         border-radius:2px;
                         font-family:serif;                                           
                         line-height:60px;
                         padding-left:17px;
                         opacity:0.8'>
                         <b>Label: {labels[counter]}</b>
                         </style>
                         <br></p>"""
    st.markdown(htmlstr1, unsafe_allow_html=True)


def get_other_candidates(sorted_entity_counts):
    #  Returns the list of promising candidates; Which have counts of greater than 1.
    other_candidates = []
    for entity in sorted_entity_counts:
        if entity[1] > 1:
            other_candidates.append(entity[0])

    return other_candidates


def extract_entities(doc):
    """
    Simply take into account those extracted entities that are not of the types {DATE, CARDINAL, ORDINAL}.
    """

    msg.info("Phase: Entities Extraction")

    entities = list()

    if doc.ents:
        for ent in doc.ents:
            print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
            if ent.label_.__eq__("DATE") or ent.label_.__eq__("ORDINAL") or ent.label_.__eq__("CARDINAL"):
                continue
            entities.append(ent.text)

    return entities


def predict_protagonists(entities_extracted):
    msg.info("Phase: Protagonists Prediction")
    predicted_protagonists = []

    entity_counts = Counter(entities_extracted)  # <Key: Entity, Value: #TimesItHasBeenMentioned-#Counts>
    most_common_entity = max(entity_counts, key=entity_counts.get)  # Get the Key/Entity with the most counts.

    predicted_protagonists.append(most_common_entity)

    # 2nd Approach which takes into account the other promising candidates.
    if entity_counts.get(most_common_entity) > 1 and len(entity_counts) > 1:
        sorted_entity_counts = sorted(entity_counts.items(), key=lambda item: item[1], reverse=True)
        other_candidates = get_other_candidates(sorted_entity_counts[1:])
        print(other_candidates)
        predicted_protagonists.extend(other_candidates)

    return predicted_protagonists


def post_process(candidates):
    """Heuristics-land; where the heuristics are applied.

    The heuristics that are applied:

    1) Check the first letter of the predicted protagonist(s). The assumption behind this is: The entity(s) about whom
    the news/article is about, always/should have proper naming convention; first letter of the word is capitalised.

    2) There are cases where the middle name of the person has all small letters. so for that, I only consider the first
    and last name. e.g, Sohail van Ziahh.

    3) The predicted protagonist cannot have more than 3 words in it. (or could be 4; trial and error, we'll see.) """

    proper_candidates = []
    for candidate in candidates:
        if len(candidate.split(" ")) > 3:
            continue
        elif len(candidate.split(" ")) == 1:
            if candidate[0][0].isupper():
                proper_candidates.append(candidate)
                continue
        tokens = candidate.split(" ")
        if tokens[0][0].isupper() and tokens[-1][0].isupper():
            proper_candidates.append(candidate)

    return proper_candidates


def coref_ner():

    COREF_MODEL = st.session_state.coref_model
    NER_MODEL = st.session_state.ner_model

    doc = COREF_MODEL(bunch_of_articles[st.session_state.counter])

    htmlstr1 = f"""<p style='background-color:yellow;
                      color:black;
                      font-size:18px;
                      border-radius:2px;
                      font-family:serif;                                           
                      line-height:60px;
                      padding-left:17px;
                      opacity:0.8'>
                      <b>Input text</b></style>
                      <br></p>"""

    st.markdown(htmlstr1, unsafe_allow_html=True)

    st.markdown(bunch_of_articles[st.session_state.counter])

    write_title_and_label(st, st.session_state.counter)

    #  st.write(titles[st.session_state.counter])

    # Print out clusters
    msg.info("Found clusters")
    for cluster in doc.spans:
        print(f"{cluster}: {doc.spans[cluster]}")

    dic = {}
    for cluster in doc.spans:
        dic[doc.spans[cluster]] = doc.spans[cluster].__len__()
    try:
        res = sorted(dic.items(), key=lambda x: x[1], reverse=True)[0][0].__getitem__(0)

        st.write(f"2nd approach: {res}")

    except:
        st.write(f"Error in 2nd approach!")
        print(f"Error for {doc}")

    updated_text = resolve_references(doc)

    msg.info("Document with resolved references")
    print(updated_text)

    entities_extracted = extract_entities(NER_MODEL(updated_text))

    if len(entities_extracted) == 0:
        msg.fail('No named entities found.')
        st.error("Something bad happened! Sorry")
        return

    predicted_protagonists = predict_protagonists(entities_extracted)

    # Post-processing
    predicted_protagonists = post_process(predicted_protagonists)

    if len(predicted_protagonists) == 0:
        msg.fail('Unable to predict the right protagonists.')
        st.error("Something bad happened! Sorry")

    else:
        print(predicted_protagonists)

        htmlstr1 = f"""<p style='background-color:green;
                                                      color:white;
                                                      font-size:18px;
                                                      border-radius:2px;
                                                      font-family:serif;                                           
                                                      line-height:60px;
                                                      padding-left:17px;
                                                      opacity:0.8'>
                                                      Predcition: Text is about <b>{predicted_protagonists}</b></style>
                                                      <br></p>"""
        st.markdown(htmlstr1, unsafe_allow_html=True)


bunch_of_articles = []
titles = []
labels = []


def coref_main():
    col1, col2 = st.columns(2)

    if 'counter' not in st.session_state:
        st.session_state.counter = 0

    show_btn = col1.button("Next Article ⏭️", on_click=on_button_click, args=([]))


def read_data():
    df = pd.read_csv("<NEWS DATA.CSV or whatever")
    df[df['labels'].notnull()].reset_index()
    df = df.sample(frac=1).reset_index(drop=True)

    #    titles.extend(list(df['title'].values))
    labels.extend(list(df['labels'].values))
    bunch_of_articles.extend(list(df['cleaned_news'].values))


def init_models():
    start = time.monotonic()

    if 'ner_model' not in st.session_state:
        st.session_state.ner_model = spacy.load("en_core_web_trf")

    if 'coref_model' not in st.session_state:
        st.session_state.coref_model = spacy.load("en_coreference_web_trf")

    print(f"Time taken: {time.monotonic() - start}")



if __name__ == '__main__':
    read_data()

    init_models()

    coref_main()
