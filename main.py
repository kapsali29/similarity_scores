from process_data import find_line_unigrams_bow, read_input, build_feature_matrix, calculate_tfidf, _similarity

file = input("Please select file to read: ")
lines = read_input(file)
uni, bows, features = find_line_unigrams_bow(lines)
feature_matrix = build_feature_matrix(bows, features)
tfidf_matrix = calculate_tfidf(feature_matrix)
_similarity(tfidf_matrix)
