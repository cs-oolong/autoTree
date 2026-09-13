from prompts.prompt_annotate_dialogs import ANNOTATE_DIALOGS_SYSTEM, ANNOTATE_DIALOGS_USER
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import argparse
import os
from dotenv import load_dotenv
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dialogs_path", type=str, help="Path to input dialogs file")
parser.add_argument("-t", "--tree_path", type=str, help="Path to decision tree file")
parser.add_argument("-m", "--model", type=str, default="gpt-4o", help="OpenAI model to use")
parser.add_argument("-b", "--binary", type=bool, default=False, help="Whether to use binary tree")
parser.add_argument("-o", "--output_dir", type=str, help="Output directory for results")
args = parser.parse_args()

dialogs_path = args.dialogs_path
tree_path = args.tree_path
model = args.model
binary = args.binary
output_dir = args.output_dir

# Set up LLM chain
prompt_annotator = ChatPromptTemplate.from_messages(
    [("system", ANNOTATE_DIALOGS_SYSTEM), ("human", ANNOTATE_DIALOGS_USER)]
)

llm_annot = ChatOpenAI(model=model, temperature=0.4)
chain_annotator = LLMChain(llm=llm_annot, prompt=prompt_annotator)


def get_question(current_node: Dict[str, Any], path: List[str]) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Get the current question and possible answers based on position in decision tree.

    Args:
        current_node: Current node in the decision tree
        path: List of keys defining path through tree to current position

    Returns:
        Tuple containing:
        - Current node dictionary
        - Question string to ask
        - List of possible answer strings
    """
    # Traverse to current node
    if path:
        for key in path:
            current_node = current_node[key]

    # Get possible answers for current node
    possible_answers = [
        f"Answer {i+1}: {current_node['groups'][i]['label']}" for i in range(len(current_node["groups"]))
    ]
    return current_node, current_node["question_to_define_groups"], possible_answers


def iterate_over_questions_binary(
    tree: Dict[str, Any], path: List[str], previous_context: str, current_utterance: str
) -> int:
    """
    Recursively traverse binary decision tree to classify utterance.

    Args:
        tree: Full decision tree dictionary
        path: Current path through tree
        previous_context: Previous dialog context
        current_utterance: Utterance to classify

    Returns:
        Integer label classification
    """
    current_node, question, possible_answers = get_question(tree, path)
    llm_inputs = {
        "previous_context": previous_context,
        "current_utterance": current_utterance,
        "question": question,
        "possible_answers": possible_answers,
    }

    # Try classification up to 2 times
    for _ in range(2):
        output = chain_annotator.invoke(llm_inputs)
        output = output.get("text")

        # Handle Answer 1
        if "answer 1" in output.lower():
            if current_node.get("next_split_group_1", {}):
                path.append("next_split_group_1")
                return iterate_over_questions_binary(tree, path, previous_context, current_utterance)
            logger.info(f"Group 1: {current_node['group_1_data']}")
            return current_node["group_1_data"][0]

        # Handle Answer 2
        elif "answer 2" in output.lower():
            if current_node.get("next_split_group_2", {}):
                path.append("next_split_group_2")
                return iterate_over_questions_binary(tree, path, previous_context, current_utterance)
            logger.info(f"Group 2: {current_node['group_2_data']}")
            return current_node["group_2_data"][0]

    # If no valid answer after retries, return default
    logger.info("Warning: No valid classification found")
    return current_node["group_2_data"][0]


def iterate_over_questions_non_binary(
    tree: Dict[str, Any], path: List[str], previous_context: str, current_utterance: str
) -> int:
    """
    Recursively traverse non-binary decision tree to classify utterance.

    Args:
        tree: Full decision tree dictionary
        path: Current path through tree
        previous_context: Previous dialog context
        current_utterance: Utterance to classify

    Returns:
        Integer label classification
    """
    current_node, question, possible_answers = get_question(tree, path)
    llm_inputs = {
        "previous_context": previous_context,
        "current_utterance": current_utterance,
        "question": question,
        "possible_answers": possible_answers,
    }

    output = chain_annotator.invoke(llm_inputs)
    output = output.get("text")

    # Check each possible answer
    for i in range(len(possible_answers)):
        if f"answer {i+1}" in output.lower():
            if current_node["groups"][i].get("next_split", {}):
                path.extend(["groups", i, "next_split"])
                return iterate_over_questions_non_binary(tree, path, previous_context, current_utterance)
            logger.info(f"Group {i+1}: {current_node['groups'][i]['data']}")
            return current_node["groups"][i]["data"][0]

    # If no valid answer found
    logger.info("Warning: No valid classification found")
    return current_node["groups"][0]["data"][0]


def output_tsv_path() -> Path:
    return Path(output_dir) / "dialogs_annotated.tsv"


def save_checkpoint(dialogs: pd.DataFrame) -> None:
    """Write progress atomically so partial runs survive API failures."""
    path = output_tsv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tsv.tmp")
    dialogs.to_csv(tmp, index=False, sep="\t")
    tmp.replace(path)


def load_dialogs_file() -> pd.DataFrame:
    if dialogs_path.endswith(".csv"):
        return pd.read_csv(dialogs_path)
    if dialogs_path.endswith(".json"):
        return pd.read_json(dialogs_path)
    if dialogs_path.endswith(".tsv"):
        return pd.read_csv(dialogs_path, sep="\t")
    raise ValueError(f"Invalid file type: {dialogs_path}")


def prepare_checkpoint(dialogs: pd.DataFrame) -> Tuple[pd.DataFrame, int, Optional[Any], Optional[str], Optional[str]]:
    """
    Resume from an existing partial output file when row counts match.
    Returns the working dataframe, next row index, and dialog context state.
    """
    path = output_tsv_path()
    if not path.exists():
        working = dialogs.copy()
        working["Annotations"] = pd.NA
        return working, 0, None, None, None

    checkpoint = pd.read_csv(path, sep="\t")
    if len(checkpoint) != len(dialogs):
        logger.warning("Checkpoint row count does not match input dialogs; starting fresh.")
        working = dialogs.copy()
        working["Annotations"] = pd.NA
        return working, 0, None, None, None

    if "Annotations" not in checkpoint.columns:
        checkpoint["Annotations"] = pd.NA

    completed = checkpoint["Annotations"].notna().sum()
    if completed == len(checkpoint):
        logger.info("All %s utterances already annotated; nothing to do.", len(checkpoint))
        return checkpoint, len(checkpoint), None, None, None

    if completed:
        logger.info("Resuming from utterance %s/%s.", completed + 1, len(checkpoint))
        prev = checkpoint.iloc[completed - 1]
        return checkpoint, completed, prev["dialog_id"], prev["speaker"], prev["text"]

    checkpoint["Annotations"] = pd.NA
    return checkpoint, 0, None, None, None


def main() -> None:
    """
    Main function to process dialog file and generate annotations.
    Reads input dialogs, processes each utterance through decision tree,
    and saves annotated results incrementally after each utterance.
    """
    dialogs = load_dialogs_file()
    working, start_idx, dialog_id_prev, previous_speaker, previous_text = prepare_checkpoint(dialogs)

    # Load decision tree
    with open(tree_path, "r") as f:
        questions_tree = json.load(f)

    for idx in range(start_idx, len(working)):
        utt = working.iloc[idx]
        dialog_id = utt["dialog_id"]
        logger.info(f"Dialog ID: {dialog_id}")

        speaker = utt["speaker"]
        text = utt["text"]
        path = []
        logger.info(f"{speaker}: {text}")

        # Set previous context
        if dialog_id_prev != dialog_id:
            previous_context = "There is no previous context, as this is a beginning of the dialog."
            dialog_id_prev = dialog_id
        else:
            previous_context = f"Speaker {previous_speaker}: {previous_text}"
            

        # Get initial question
        _, question, possible_answers = get_question(questions_tree, [])
        llm_inputs = {
            "previous_context": previous_context,
            "current_utterance": f"Speaker {speaker}: {text}",
            "question": question,
            "possible_answers": possible_answers,
        }

        # Get initial classification
        output = chain_annotator.invoke(llm_inputs)
        output = output.get("text")

        # Set path based on classification
        for i in range(len(possible_answers)):
            if f"answer {i+1}" in output.lower():
                if not binary:
                    if questions_tree["groups"][i].get("next_split", {}):
                        path.extend(["groups", i, "next_split"])
                    break  # Add break to prevent further iterations
                else:
                    if i == 0:  # answer 1
                        if questions_tree.get("next_split_group_1", {}):
                            path.append("next_split_group_1")
                    elif i == 1:  # answer 2
                        if questions_tree.get("next_split_group_2", {}):
                            path.append("next_split_group_2")
                    break  # Add break to prevent further iterations

        current_utterance = f"Speaker {speaker}: {text}"
        if binary:
            label = iterate_over_questions_binary(questions_tree, path, previous_context, current_utterance)
        else:
            label = iterate_over_questions_non_binary(questions_tree, path, previous_context, current_utterance)

        # Update previous utterance info
        previous_speaker = speaker
        previous_text = text
        working.at[idx, "Annotations"] = label
        save_checkpoint(working)
        logger.info("Saved checkpoint: %s/%s utterances annotated.", idx + 1, len(working))


if __name__ == "__main__":
    main()
