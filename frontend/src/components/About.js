export default function About() {
  return (
    <div className="panel about">
      <h2>How it works</h2>
      <ol>
        <li><strong>Clean-up.</strong> Disguised spellings are undone (<code>l@nja</code> → <code>lanja</code>, <code>gooood</code> → <code>good</code>), links and @handles are removed.</li>
        <li><strong>Word list.</strong> A message with a severe abusive word and no earlier messages is flagged straight away. Milder or ambiguous words are only noted.</li>
        <li><strong>Model.</strong> Everything else is read by a fine-tuned IndicBERT model together with up to three earlier messages. It scores five kinds of harm: toxic, severe, identity attack, insult and threat.</li>
        <li><strong>Decision.</strong> A message is flagged when its score passes a line tuned on validation data. Scores close to that line are marked <em>Needs review</em>.</li>
      </ol>
      <h2>Limits</h2>
      <ul>
        <li>The Telugu training data is small; Tamil code-mixed comments and translated prompts fill the gap. Expect mistakes on slang the model has not seen.</li>
        <li>The highlighted words show where the model&apos;s attention went, which is a hint, not a proof of why it decided.</li>
        <li>Use it to support a human moderator, not to replace one.</li>
      </ul>
    </div>
  );
}
