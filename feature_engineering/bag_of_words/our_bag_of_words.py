def single_sentence_bag_of_words(sentence: str) -> dict[str, int]:
    """
    Takes in one sentence, and returns a bag of words dictionary for the sentence

    a bag of words dictionary for a sentence
    contains the word (in all passed sentences) as the key, and the value as the count in each sentence.
    """
    # get bag of words for sentence
    bag_of_words: dict[str, int] = {}
    words: list[str] = sentence.split()
    for word in words:
        bag_of_words[word] = bag_of_words.get(word, 0) + 1
    return bag_of_words


def multiple_sentence_bag_of_words(sentences: list[str]) -> list[dict[str, int]]:
    """
    Takes in a list of sentences, and returns a bag of words dictionary for each sentence.

    a bag of words dictionary for a sentence
    contains the word (in all passed sentences) as the key, and the value as the count in each sentence.
    """
    # keep track of words in all sentences
    all_words: set[str] = set()

    # Enrich default_bag_of_words with all words in all sentences, and set them to 0
    for sentence in sentences:
        words: list[str] = sentence.split()
        for word in words:
            all_words.add(word)

    bag_of_words: list[dict[str, int]] = []

    # get a bag of words for each sentence
    for sentence in sentences:
        current_bag_of_words: dict[str, int] = {}
        # get bag of words for sentence
        words: list[str] = sentence.split()
        for word in words:
            current_bag_of_words[word] = current_bag_of_words.get(word, 0) + 1

        # add words found in other sentences into the same dictionary, set to 0
        for word in all_words:
            current_bag_of_words[word] = current_bag_of_words.get(word, 0)
        bag_of_words.append(current_bag_of_words)
    return bag_of_words


if __name__ == "__main__":
    sentence_one: str = "I love sunflower seeds"
    sentence_two: str = "I hate millet seeds"

    # first_bag_of_words: {'I': 1, 'love': 1, 'sunflower': 1, 'seeds': 1}
    first_bag_of_words: dict[str, int] = single_sentence_bag_of_words(sentence_one)
    print(f"first_bag_of_words: {first_bag_of_words}")

    # second_bag_of_words: {'I': 1, 'hate': 1, 'millet': 1, 'seeds': 1}
    second_bag_of_words: dict[str, int] = single_sentence_bag_of_words(sentence_two)
    print(f"second_bag_of_words: {second_bag_of_words}")

    """
    list_of_bag_of_words: [
        {'I': 1, 'love': 1, 'sunflower': 1, 'seeds': 1, 'hate': 0, 'millet': 0}, 
        {'I': 1, 'love': 0, 'sunflower': 0, 'seeds': 1, 'hate': 1, 'millet': 1}
    ]
    """
    list_of_bag_of_words: list[dict[str, int]] = multiple_sentence_bag_of_words(
        [sentence_one, sentence_two]
    )
    print(f"list_of_bag_of_words: {list_of_bag_of_words}")
