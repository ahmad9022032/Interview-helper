"use client";

import type { Stage } from "@/lib/types";

interface Props {
  question: string;
  answer: string;
  stage: Stage;
  truncated: boolean;
}

export function Answer({ question, answer, stage, truncated }: Props) {
  const streaming = stage === "answering" || stage === "thinking";
  return (
    <section className="panel panel-answer">
      <h2>Suggested answer</h2>
      {question ? (
        <div className="question">
          Question detected: <strong>{question}</strong>
        </div>
      ) : (
        <div className="question">Waiting for a question...</div>
      )}
      {answer ? (
        <div className={`answer ${streaming ? "caret" : ""}`}>{answer}</div>
      ) : stage === "thinking" ? (
        <div className="answer muted caret">Generating answer</div>
      ) : (
        <div className="answer muted">
          Start listening, then let the interviewer talk. Their question and your answer appear here.
        </div>
      )}
      {truncated && !streaming && (
        <div className="note">Cut off at the token limit. Raise MAX_ANSWER_TOKENS in .env for longer answers.</div>
      )}
    </section>
  );
}
