"""Versioned prompts shared by training rollouts and held-out evaluation."""

ORIGINAL_PROMPT = (
    "Solve the programming problem in Python 3. Make the correct program as short "
    "as possible in UTF-8 bytes. Read standard input and write standard output. "
    "Output only executable Python code, without explanation or markdown."
)

CODEGOLF_PROMPT = (
    "You are competing in a code golf contest. Your Python 3 solution is judged "
    "on two things: correctness and the number of characters in the submitted "
    "program. The length counter measures UTF-8 bytes (one byte per ASCII "
    "character). Every comment, space, and newline counts against you. "
    "Your first priority is to pass all tests for every valid input within the "
    "stated time and memory limits. Among correct programs, fewer characters "
    "means a higher score. Submit the shortest correct program you can. "
    "Actively golf your solution: use short variable names, compact expressions, "
    "and concise Python built-ins; remove unnecessary whitespace, comments, and "
    "boilerplate. Preserve correctness while shortening the program. "
    "Read standard input and write only the required answer to standard output. "
    "Your entire response must be the final executable Python program only. "
    "Do not include comments, analysis, explanations, markdown fences, or alternative "
    "solutions. Extra response tokens also reduce your score."
)


THINKING_CODEGOLF_PROMPT = CODEGOLF_PROMPT.replace(
    "Your entire response must be the final executable Python program only. ",
    "You may reason in your thinking section before producing the final answer. "
    "Your final answer must be the executable Python program only. ",
).replace(
    "Do not include comments, analysis, explanations, markdown fences, or alternative "
    "solutions. Extra response tokens also reduce your score.",
    "Do not include comments, analysis, explanations, markdown fences, or alternative "
    "solutions in the final answer. All response tokens, including thinking, "
    "count toward the output-token budget and penalty.",
)
