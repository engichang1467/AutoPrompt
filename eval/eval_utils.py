import json
from estimator.estimator_llm import LLMEstimator


def set_function_from_iterrow(func):
    def wrapper(dataset):
        dataset['score'] = dataset.apply(func, axis=1)
        return dataset

    return wrapper



def set_ranking_function(params):
    evaluator = LLMEstimator(params)
    evaluator.init_chain(params.label_schema)
    evaluator.mode = 'score'

    def wrapper(dataset):
        generation_dataset = dataset.copy()
        generation_dataset['text'] = '###User input:\n' + generation_dataset['text'] + '\n####model prediction:\n' + generation_dataset['prediction']

        # Apply the evaluation function
        generation_dataset = evaluator.apply_dataframe(generation_dataset)
        
        # Collect scores and reasonings directly in the main body
        scores_with_reasoning_list = []
        for index, row in generation_dataset.iterrows():
            scores_with_reasoning = {
                "hallucination_score": {"score": row.get("hallucination_score", 0), "reasoning": row.get("hallucination_reasoning", "No reasoning provided")},
                "classification_score": {"score": row.get("classification_score", 0), "reasoning": row.get("classification_reasoning", "No reasoning provided")},
                "aggregation_score": {"score": row.get("aggregation_score", 0), "reasoning": row.get("aggregation_reasoning", "No reasoning provided")},
                "response_quality_score": {"score": row.get("response_quality_score", 0), "reasoning": row.get("response_quality_reasoning", "No reasoning provided")},
            }
            scores_with_reasoning_list.append(json.dumps(scores_with_reasoning))

        generation_dataset['score'] = scores_with_reasoning_list

        dataset['score'] = generation_dataset['score']
        return dataset

    return wrapper





# to do: 
- Any downstream code that reads the score field from the dataset now needs to parse the JSON string to access individual scores and reasonings.
    
