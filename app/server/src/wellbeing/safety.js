// Safety layer. Pure code, no LLM: it runs on every user message BEFORE any model reply is shown.
//
// detectSafety(text) -> { level: 'none' | 'possible' | 'explicit', matches: [ids] }
//   explicit : a first-person statement of suicide / self-harm intent or a strong hopelessness phrase
//   possible : softer hopelessness, or a negated mention ("I'm not suicidal")
// Idioms such as "this exam is killing me" must not trigger. When unsure the detector errs on the side
// of triggering, because a false alarm costs a short supportive message and a miss costs much more.

const NEG_BEFORE = /\b(?:not|never|don'?t|dont|do\s+not|didn'?t|no|won'?t|isn'?t|wouldn'?t|aren'?t|wasn'?t|nahi|nahin|nhi)\b[^.!?]{0,25}$/i;
const NEG_AFTER = /^[^.!?]{0,15}\b(?:nahi|nahin|nhi|not)\b/i;

// negatable: a negation near the match turns an explicit hit into 'possible' (a soft mention).
const EXPLICIT = [
  // --- English: intent to die or self-harm ---
  { id: 'kill_myself', negatable: true, re: /\b(?:kill|killing|hang|hanging|drown|drowning|shoot|shooting)\s+my\s?self\b/gi, exclude: /^\s*(?:laughing|laughter|with\s+laughter|from\s+laughing)/i },
  { id: 'end_my_life', negatable: true, re: /\b(?:end|ending|take|taking)\s+my\s+(?:own\s+)?life\b/gi },
  { id: 'end_it_all', negatable: true, re: /\b(?:want|wanna|going|gonna|plan(?:ning)?|thinking|ready)\s+(?:to\s+|of\s+|about\s+)?(?:end|ending)\s+(?:it\s+all|everything)\b/gi },
  { id: 'want_to_die', negatable: true, re: /\b(?:want(?:ed)?|wanna|ready|planning|need)\s+(?:to\s+)?die\b/gi, exclude: /^\s*(?:of|from|laughing|inside|with|a\s+little)\b/i },
  { id: 'wish_dead', negatable: true, re: /\bwish\s+(?:i|that\s+i)\s+(?:was|were|could\s+be|would\s+be)\s+dead\b/gi },
  { id: 'better_off_dead', negatable: false, re: /\bbetter\s+off\s+dead\b/gi },
  { id: 'dont_want_to_live', negatable: false, re: /\b(?:don'?t|dont|do\s+not|no\s+longer)\s+want\s+to\s+(?:live|be\s+alive|be\s+here|exist|wake\s+up)\b/gi },
  { id: 'self_harm_intent', negatable: true, re: /\b(?:want|wanna|going|gonna|need|urge|thinking|started|keep|been|plan(?:ning)?|feel\s+like)\b[^.!?]{0,25}\b(?:hurt|harm|cut|cutting|injure|punish|burn|burning)\w*\s+my\s?self\b/gi },
  { id: 'cutting_myself', negatable: true, re: /\b(?:cutting|burning)\s+myself\b/gi },
  { id: 'self_harm_term', negatable: true, re: /\b(?:i|i'?m|im|i\s+have|i'?ve|ive)\b[^.!?]{0,20}\bself[-\s]?harm(?:ing|ed)?\b/gi },
  { id: 'kms', negatable: false, re: /\bkms\b/gi },
  // --- strong hopelessness ---
  { id: 'no_reason_to_live', negatable: false, re: /\b(?:no\s+(?:reason|point)\s+(?:to|in)\s+(?:live|living|going\s+on|being\s+alive|life)|nothing\s+to\s+live\s+for|not\s+worth\s+living|life\s+(?:is\s+)?not\s+worth|can'?t\s+go\s+on\s+(?:living|like\s+this\s+any\s*more))\b/gi },
  { id: 'better_off_without_me', negatable: false, re: /\b(?:better\s+off\s+without\s+me|no\s+one\s+(?:would|will)\s+(?:miss|notice|care\s+if)\s+(?:me|i))\b/gi },
  { id: 'tired_of_living', negatable: false, re: /\b(?:tired|sick|sick\s+and\s+tired)\s+of\s+living\b/gi },
  // --- Hinglish (Roman script, common spellings) ---
  { id: 'hi_mar_jana_a', negatable: true, re: /\bmar\s*jaa?(?:na|un|u|oon|aun)\b[^.!?]{0,20}\b(?:chah\w*|mann?|dil|soch\w*|behtar|accha|acha|better)\b/gi, postNeg: true },
  { id: 'hi_mar_jana_b', negatable: true, re: /\b(?:chah\w*|mann?|dil)\b[^.!?]{0,12}\bmar\s*jaa?(?:na|un|u|oon|aun)\b/gi, postNeg: true },
  { id: 'hi_marna', negatable: true, re: /\bmar(?:na|ne)\s+(?:chah\w*|ka\s+(?:mann?|dil|man))\b/gi, postNeg: true },
  { id: 'hi_jeena_nahi', negatable: false, re: /\b(?:mujhe\s+)?jee?na\s+(?:hi\s+)?(?:nahi|nahin|nhi)\s+(?:chah\w*|hai|h)\b/gi },
  { id: 'hi_jeene_ka_mann_nahi', negatable: false, re: /\bjee?ne\s+(?:ka|ki)\s+(?:mann?|man|dil|ichh?a|iccha|wajah|vajah|maqsad|umeed)\s+(?:hi\s+)?(?:nahi|nahin|nhi)\b/gi },
  { id: 'hi_jeene_ka_matlab', negatable: false, re: /\bjee?ne\s+ka\s+koi\s+(?:matlab|maksad|wajah|vajah)\b/gi },
  { id: 'hi_zindagi_khatam', negatable: false, re: /\b(?:zindagi|jeevan|life)\s+(?:khatam|khtm|khtam)\s+(?:kar|kr)\w*/gi },
  { id: 'hi_khud_ko', negatable: false, re: /\bkhud\s*ko\s+(?:khatam|maar|mar|hurt|nuksan)\b/gi },
  { id: 'hi_jaan_dena', negatable: false, re: /\b(?:apni\s+)?jaan\s+(?:de|dena|lena|le)\s*(?:du|dun|dunga|dungi|doon|dene|lene)?\b[^.!?]{0,10}\b(?:chah\w*|mann?|dil)\b|\bjaan\s+de\s+(?:du|dun|dunga|dungi|doon)\b/gi },
  { id: 'hi_khudkushi', negatable: false, re: /\b(?:khud\s*kushi|khudkushi|aatm\s*hatya|aatmahatya|aatmghat)\b/gi },
  { id: 'hi_suicide_kar', negatable: false, re: /\b(?:suicide|sucide|suside)\s+(?:kar|kr)\w*/gi },
  { id: 'hi_jee_ke_kya', negatable: false, re: /\bjee?\s+ke\s+kya\s+kar(?:na|u|un)\b/gi },
];

const POSSIBLE = [
  { id: 'hopeless', re: /\b(?:hopeless|helpless|worthless|pointless|no\s+way\s+out|can'?t\s+take\s+(?:it|this)\s+any\s*more|can'?t\s+do\s+this\s+any\s*more|nothing\s+matters|(?:i'?m|i\s+am|im)\s+(?:a\s+)?burden|burden\s+(?:to|on)\s+everyone|tired\s+of\s+(?:everything|life)|done\s+with\s+(?:everything|life)|no\s+future|lost\s+all\s+hope|give\s+up\s+on\s+(?:life|everything)|disappear\s+forever|sab\s+khatam|koi\s+umeed\s+nahi|kuch\s+nahi\s+bacha|thak\s+(?:gaya|gayi)\s+hu+n?\s+(?:zindagi|life))\b/gi },
];

// Suicide words are matched per sentence: a first-person word must be present and no "studying about it" word.
const SUICIDE_WORD = /\b(?:suicid\w*|sucid\w*|suside|sucide)\b/i;
const FIRST_PERSON = /\b(?:i|i'?m|im|i'?ve|ive|my|me|myself|mujhe|mera|meri|main|mai)\b/i;
const ACADEMIC = /\b(?:read|reading|article|news|lecture|documentary|movie|film|show|prevention|awareness|statistics|studying|syllabus|essay|assignment|project|topic|research)\b/i;

function negated(text, start, end, spec) {
  if (NEG_BEFORE.test(text.slice(Math.max(0, start - 30), start))) return true;
  return !!spec.postNeg && NEG_AFTER.test(text.slice(end));
}

export function detectSafety(input) {
  const text = String(input ?? '').normalize('NFKC').replace(/[’‘]/g, "'");
  const hits = { explicit: [], possible: [] };

  for (const spec of EXPLICIT) {
    spec.re.lastIndex = 0;
    let m;
    while ((m = spec.re.exec(text)) !== null) {
      const end = m.index + m[0].length;
      if (spec.exclude && spec.exclude.test(text.slice(end))) continue;
      if (spec.negatable && negated(text, m.index, end, spec)) hits.possible.push(`${spec.id}:negated`);
      else hits.explicit.push(spec.id);
    }
  }

  for (const sentence of text.split(/(?<=[.!?\n])\s+/)) {
    if (SUICIDE_WORD.test(sentence) && FIRST_PERSON.test(sentence) && !ACADEMIC.test(sentence)) {
      const m = SUICIDE_WORD.exec(sentence);
      if (negated(sentence, m.index, m.index + m[0].length, {})) hits.possible.push('suicide_word:negated');
      else hits.explicit.push('suicide_word');
    }
  }

  for (const spec of POSSIBLE) {
    spec.re.lastIndex = 0;
    if (spec.re.test(text)) hits.possible.push(spec.id);
  }

  if (hits.explicit.length) return { level: 'explicit', matches: [...new Set(hits.explicit)] };
  if (hits.possible.length) return { level: 'possible', matches: [...new Set(hits.possible)] };
  return { level: 'none', matches: [] };
}

// Highest of two levels (used to combine the regex detector with the structured Gemini signal).
const RANK = { none: 0, possible: 1, explicit: 2 };
export function combineSignals(...levels) {
  return levels.reduce((a, b) => (RANK[b] > RANK[a] ? b : a), 'none');
}

export const HELPLINES = [
  { name: 'Tele-MANAS (free, 24x7, India)', number: '14416' },
  { name: 'AASRA', number: '9820466726' },
  { name: 'Emergency services', number: '112' },
];

// Fixed, pre-written message. Never generated by an LLM.
export function safetyMessage({ counsellorContact, repeat = false } = {}) {
  const contact = counsellorContact || 'your college counselling centre';
  const lines = HELPLINES.map((h) => `- ${h.name}: ${h.number}`).join('\n');
  if (repeat) {
    return (
      `I'm still here with you, and I'm taking what you said seriously. If you might act on these thoughts, ` +
      `please call 112 now or reach out to someone you trust in person.\n${lines}\n` +
      `You can also contact ${contact}. You don't have to carry this alone.`
    );
  }
  return (
    `Thank you for telling me this. It sounds like you are in a lot of pain right now, and I'm really glad you said it out loud. ` +
    `I'm an automated assistant, not a counsellor, and I can't promise to keep this private or to keep you safe myself, ` +
    `so please reach out to a real person right now.\n\n` +
    `If you are in immediate danger or might act on these thoughts, call emergency services (112). You can also talk to someone for free, any time:\n` +
    `${lines}\n\n` +
    `Please also tell a person you trust, such as a friend, family member or hostel warden, and contact ${contact} today. ` +
    `You don't have to go through this alone. We can keep talking here too, and I'll stay with you.`
  );
}

// One short fixed line appended to a normal reply when the signal is only "possible".
export function softSupportLine({ counsellorContact } = {}) {
  return `If things ever feel too heavy, you can talk to Tele-MANAS (14416, free, 24x7) or ${counsellorContact || 'your college counselling centre'}.`;
}

// Output guard: a model reply that discusses methods of self-harm is never shown.
const METHOD_WORDS = /\b(?:overdos\w*|lethal\s+dose|hang(?:ing)?\s+(?:yourself|myself|from)|noose|slit(?:ting)?\s+(?:your|my)\s+wrists?|jump(?:ing)?\s+(?:off|from)\s+(?:a\s+)?(?:bridge|building|roof)|how\s+to\s+(?:kill|end\s+your))/i;
export function replyIsUnsafe(text) {
  return METHOD_WORDS.test(String(text ?? ''));
}

// Output guard 2: the assistant must never label the user with a condition ("you have depression", "you are bipolar").
const DIAGNOSIS = /\byou(?:'re|\s+are|\s+have|\s+seem\s+to\s+have|\s+might\s+have|\s+may\s+have|\s+probably\s+have|\s+likely\s+have|\s+sound\s+(?:like\s+you\s+have|depressed))\b[^.!?]{0,30}\b(?:depress\w*|anxiety\s+disorder|bipolar|ptsd|clinically|mentally\s+ill|a\s+disorder|suicidal)\b/i;
export function replyHasDiagnosis(text) {
  return DIAGNOSIS.test(String(text ?? ''));
}