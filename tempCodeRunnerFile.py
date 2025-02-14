preprocessed_review = preprocess_text(data['Content'].values)
# data['Content'] = preprocessed_review

# consolidated = ' '.join(
#     word for word in data['Content'][data['Label'] == 1].dropna().astype(str)
# )
# if len(consolidated) > 0:
#     wordCloud = WordCloud(
#         width=1600, height=800, random_state=21, max_font_size=110, collocations=False
#     )
#     plt.figure(figsize=(15, 10))
#     plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
#     plt.axis('off')
#     plt.show()
# else:
#     print("No words available for the Word Cloud.")