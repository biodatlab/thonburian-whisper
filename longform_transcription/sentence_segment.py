import os
import numpy as np
from ssg import syllable_tokenize
from attacut import tokenize
from utils import max_intervals, sum_index_list, is_english
from typing import NamedTuple, List, Dict
from dataclasses import dataclass


class SyllableRanges(NamedTuple):
    start: int
    end: int


@dataclass
class SyllableSegmentation:
    """
    A class to segment a given sentence based on the number of syllables.
    """

    def count_english_syllables(self, text: str):
        """
        Count the number of syllables in an English text.

        Parameters:
        - text (str): The input English text.

        Returns:
        - int: The number of syllables.
        """
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        if text[0] in vowels:
            count += 1
        for index in range(1, len(text)):
            if text[index] in vowels and text[index - 1] not in vowels:
                count += 1
        if text.endswith("e"):
            count -= 1
        if count == 0 and text != " ":
            count += 1

        return count

    def convert_to_syllable_ranges(
        self, integer_input_list: List[int]
    ) -> List[SyllableRanges]:
        """
        Convert a list of integers to a list of SyllableRanges.

        Parameters:
        - integer_input_list (List[int]): List of integer values.

        Returns:
        - List[SyllableRanges]: List of ranges indicating the start and end of syllables.
        """
        result = []
        for i in range(len(integer_input_list) - 1):
            result.append(
                SyllableRanges(
                    int(integer_input_list[i]), int(integer_input_list[i + 1])
                )
            )
        return result

    def update_syllable_ranges(
        self,
        previous_syllable_ranges_tuples: List[SyllableRanges],
        sum_syllable_list: List[int],
    ) -> list[tuple]:
        """
        Update the syllable ranges based on previous ranges and the sum of syllable list to avoid cutting words in the middle.

        Parameters:
        - previous_syllable_ranges_tuples (List[SyllableRanges]): List of previous syllable ranges.
        - sum_syllable_list (List[int]): Cumulative sum of syllable list.

        Returns:
        - List[tuple]: Updated list of syllable ranges.
        """
        index = 0
        updated_syllable_ranges = []
        for syllable_range in previous_syllable_ranges_tuples:
            if not updated_syllable_ranges:
                start = 0
            else:
                start = sum_syllable_list[index]

            while index < len(sum_syllable_list):
                if syllable_range.end <= sum_syllable_list[index]:
                    updated_syllable_ranges.append(
                        SyllableRanges(start, sum_syllable_list[index])
                    )
                    break
                else:
                    index += 1
        return updated_syllable_ranges

    def get_syllable_index(
        self, syllable_ranges: List[SyllableRanges], sum_syllable_list: List[int]
    ) -> List[tuple]:
        """
        Retrieve the syllable indexes based on syllable ranges and the sum syllable list.

        Parameters:
        - syllable_ranges (List[SyllableRanges]): List of syllable ranges.
        - sum_syllable_list (List[int]): Cumulative sum of syllable list.

        Returns:
        - List[tuple]: List of tuples indicating word level index ranges.
        """
        word_indexes = []
        for syllable_range in syllable_ranges:
            if syllable_range.start == 0:
                start_index = 0
                end_index = sum_syllable_list.index(syllable_range.end) + 1
            else:
                start_index = sum_syllable_list.index(syllable_range.start) + 1
                end_index = sum_syllable_list.index(syllable_range.end) + 1
            word_indexes.append((start_index, end_index))
        return word_indexes

    def __call__(self, vad_transcriptions: List[Dict], segment_duration: float = 6.0):
        """
        Main callable function to segment the sentences in the input based on syllables and specified duration.

        Parameters:
        - vad_transcriptions (List[Dict]): Input list containing start, end, and prediction keys.
        - segment_duration (float): Desired duration for each segment. Defaults to 6.0.

        Returns:
        - List[Dict]: List of segmented transcriptions with start, end, and text.
        """
        assert segment_duration > 0.0, "segment_duration must be greater than 0!"
        segments = []
        for start, end, text in zip(
            vad_transcriptions["start"],
            vad_transcriptions["end"],
            vad_transcriptions["prediction"],
        ):
            duration = end - start
            if duration > segment_duration:
                words = tokenize(text)
                syllables = [
                    len(syllable_tokenize(word))
                    if not is_english(word)
                    else self.count_english_syllables(word)
                    for word in words
                ]
                num_ranges = max_intervals(duration, segment_duration)
                range_start, range_stop = 0, sum(syllables)
                ranges = np.linspace(range_start, range_stop, num_ranges + 1)
                syllable_ranges = self.convert_to_syllable_ranges(ranges)
                sum_syllables_indexes = sum_index_list(syllables)
                syllable_indexes = self.update_syllable_ranges(
                    syllable_ranges, sum_syllables_indexes
                )
                index_word_chunks = self.get_syllable_index(
                    syllable_indexes, sum_syllables_indexes
                )
                chunks = ["".join(words[start:end]) for start, end in index_word_chunks]
                time_split_durations = [
                    round((end - start) / sum(syllables) * duration, 3)
                    for start, end in syllable_indexes
                ]
                for chunk, time_split_duration in zip(chunks, time_split_durations):
                    segments.append(
                        {
                            "text": chunk,
                            "start": start,
                            "end": start + time_split_duration,
                        }
                    )
                    start += time_split_duration
            else:
                segments.append({"text": text, "start": start, "end": end})
        return segments
