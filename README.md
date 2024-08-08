# Turtle Benchmark

[中文](./README_zh.md)

Turtle Benchmark is a novel, uncheatable benchmark for evaluating Large Language Models (LLMs) based on the "Turtle Soup"(海龟汤) game, focusing on logical reasoning and contextual understanding.

### Highlights

- **Objective and Unbiased**: Eliminates the need for background knowledge, focusing purely on reasoning abilities.
- **Quantifiable Results**: Clear, measurable outcomes (correct/incorrect/unknown) for easy comparison.
- **Constantly Evolving**: Uses real user-generated questions, making it impossible to "game" the system.
- **Language Understanding**: Tests the model's ability to comprehend context and make logical inferences.

### Usage

```bash
cd evaluation

mv .env.example .env
# add API key.

# Default: 2-shot learning
python evaluate.py

# Zero-shot for faster evaluation
python evaluate.py --shot 0
```

### Data

- 32 unique "Turtle Soup" stories.
- 1537 human-annotated labels from users' questions.
- Our evaluation log.

### Results

#### 1. Overall Accuracy

The overall accuracy of each model across all test cases.

![Overall Benchmark Results](/evaluation/imgs/Turtle-Benchmark-result.png)

#### 2. Average Accuracy Across Stories

To mitigate potential bias from models performing poorly on specific stories with a large number of test samples, we calculated the average accuracy for each model across all 32 stories individually.

![Results Across 32 Stories](/evaluation/imgs/Turtle-Benchmark-over-32stories.png)

#### 3. Performance Chart

This scatter plot compares the overall accuracy (x-axis) with the average story accuracy (y-axis) for each model in the 2-shot learning scenario.

![2-Shot Learning Performance](/evaluation/imgs/average_model_accuracy_over_stories_2-shot.png)

### Interpretation

Based on these results, we can clearly see the performance differences among the various models:

1. **First Tier**: Claude 3.5 Sonnet stands out as the undisputed leader, significantly outperforming all other models.

2. **Second Tier**: GPT-4o, Qwen-2(通义千问), Moonshot AI(月之暗面), LLama3.1 405B, and Minimax form the second tier. While we've avoided further subdivisions, there's a noticeable decrease in performance within this group, following the order listed.

3. **Third Tier**: Douban(豆包), DeepSeek, and LLama3.1 70B constitute the third tier.

4. **Fourth Tier**: GPT-4o-mini stands alone in the fourth tier.

5. **Obsolete**: GPT-3.5's performance suggests it's no longer competitive in this context.

It's important to note that this evaluation specifically targets the models' Chinese language understanding and reasoning capabilities. In the future, pending resources and funding, we plan to translate all stories and test questions into English and re-run the tests using English prompts. This will help eliminate any performance discrepancies that may be attributed to language differences.

### Acknowledgments

We would like to express our gratitude to:

- **Steven Shi (石允丰)** from 5Y Capital for his generous financial support of the token usage required for this research.
- **Jerry Zhao (赵乾之)** for his invaluable assistance in annotating over 26,000 data points.

Your contributions have been instrumental in making this benchmark possible.
