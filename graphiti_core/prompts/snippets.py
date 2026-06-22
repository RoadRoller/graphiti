"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

summary_instructions = f"""Guidelines:
1. Output only factual content. Never explain what you're doing, why, or mention limitations or constraints.
2. Only use the provided messages, entity, and entity context to set attribute values.
3. Keep the summary information-dense and entity-specific. STATE FACTS DIRECTLY IN UNDER {MAX_SUMMARY_CHARS} CHARACTERS.
4. Preserve all materially relevant names, roles, places, dates, counts, and temporal qualifiers that are explicitly supported.
5. Prefer compact factual sentences over vague thematic phrasing or meta-language.
6. When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
7. Use communication verbs only when the act of speaking, asking, sharing, presenting, announcing, or telling is itself the important fact.
8. Never use filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", or "indicated" unless the communication act itself is the fact.
9. Include temporal anchors when the messages provide them and they help ground the fact.
10. Begin with the entity name or a direct fact, not with "A", "An", "The", or "This is" unless that wording is part of the entity name.

Example summary:
BAD: "The context shows John ordered pizza. Due to length constraints, other details are omitted from this summary."
GOOD: "John ordered pepperoni pizza from Mario's at 7:30 PM and had it delivered to the office."
"""

summarize_pair_instructions = f"""
IMPORTANT:
- Preserve all materially relevant names, roles, places, dates, counts, and changes over time that are explicitly supported.
- Prefer compact factual sentences over vague thematic phrasing.
- When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
- Use communication verbs only when the act of speaking, asking, sharing, presenting, or announcing is itself the important fact.
- Avoid filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", and "indicated" unless the communication act itself matters.
- SUMMARIES MUST BE LESS THAN {MAX_SUMMARY_CHARS} CHARACTERS.
"""

meta_language_ban = (
    'NEVER use meta-language verbs: "mentioned", "discussed", "noted", "stated", '
    '"described", "referenced", "indicated", "reported", "talked about", "brought up" — '
    'these describe conversational dynamics, not knowledge. State facts directly instead.'
)

meta_language_ban_saga_extra = (
    "NEVER refer to the messages, conversation, thread, or participants' communicative acts. "
    'The output must read as if no conversation happened — only the facts matter.\n'
    'NEVER begin with "This conversation", "The thread", "In this thread", or "The discussion".\n'
    'NEVER infer preferences or habits from a single passing mention. When a person '
    'explicitly states a preference ("I prefer X", "I love X", "I always do X"), '
    'capture it as a stated preference attributed to that person.'
)

output_discipline = (
    'OUTPUT DISCIPLINE:\n'
    '- Entity `name` is a literal mention from CURRENT_MESSAGES, ≤5 words. NEVER use a full\n'
    '  sentence, action item, goal statement, or quoted aspiration as a name.\n'
    '  BAD: "Establish a firm training/onboarding program",\n'
    '       "Secure competitive advantage through IP",\n'
    '       "Decide whether to expand into Europe next quarter".\n'
    '  GOOD (terse-name fallback for the same source content): "training program",\n'
    '       "competitive advantage", "European expansion" — extract the topical noun\n'
    '       phrase, not the full proposition. Multi-word topic names like "watercolor\n'
    '       painting" or "VR gaming" remain valid (see ENTITY RULE 4).\n'
    '- The `fact` field is one self-contained sentence. NEVER include reasoning, hedging\n'
    '  ("appears to", "implies", "suggests"), parenthetical commentary, or schema-description text.\n'
    '- `relation_type` is SCREAMING_SNAKE_CASE letters/underscores only. NEVER spaces,\n'
    '  punctuation, or sentences.\n'
    '- Output ONLY the JSON specified by the response schema. No preamble, no trailing notes,\n'
    '  no explanation of choices.'
)

_entity_exclusion_base = """
- Abstract concepts or feelings (joy, balance, growth, resilience, happiness, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  way, tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Bare relational or kinship terms (dad, mom, mother, father, sister, brother, husband, wife,
  spouse, son, daughter, uncle, aunt, cousin, grandma, grandpa, friend, boss, teacher, neighbor,
  roommate) and bare animal/pet words (dog, cat, pet, puppy, kitten). These are too generic on
  their own. Instead, qualify them with the possessor: extract "Nisha's dad" not "dad",
  "Jordan's dog" not "dog".
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")
"""

entity_exclusion_list_conversational = f"""
NEVER extract any of the following:
- Pronouns (you, me, I, he, she, they, we, us, it, them, him, her, this, that, those)
{_entity_exclusion_base.strip()}
- Sentence fragments or clauses ("what you really care about", "results of that effort")
- Adjectives or descriptive phrases ("amazing", "something different", "new hair color")
- Duplicate references to the same real-world entity. Extract each entity at most once per message,
  even if it appears multiple times or both as a speaker label and in the body text.
"""

entity_exclusion_list_text = """
NEVER extract any of the following:
- Pronouns (you, me, he, she, they, it, them, him, her, we, us, this, that, those)
- Abstract concepts (joy, balance, growth, resilience, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses as entity names
- Bare relational or kinship terms (dad, mom, sister, brother, spouse, friend, boss, pet, dog,
  cat) unless qualified with a possessor (e.g., "Nisha's dad" is acceptable, "dad" alone is not)
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")
"""

entity_exclusion_list_json = """
NEVER extract any of the following:
- Date, time, or timestamp values
- Abstract concepts or generic field values (e.g., "true", "active", "pending")
- Numeric IDs or codes that are not meaningful entity names
- Bare relational or kinship terms (e.g., "spouse", "parent", "pet") — only extract if qualified
  with a possessor name
- Bare generic objects or common nouns (e.g., "supplies", "tickets", "gear") — only extract if
  qualified with a distinguishing detail
- Generic media/content nouns unless uniquely identified in the value itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the value itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on surrounding text rather than the extracted value itself
"""

_attribute_hard_rules_common = """
2. NEVER write reasoning, justification, or commentary into any field. Specifically:
   - NEVER include parenthetical explanations like "(implied by ...)", "(Context: ...)",
     "(not explicitly stated ...)", "(based on ...)".
   - NEVER include first-person or deliberative phrases like "I should...", "However...",
     "Sticking to...", "Since no...", "the instruction is to...", "must be kept...",
     "if no value is present...".
   - NEVER list alternatives or candidates inside one field ("X, or Y, or maybe Z").
   - NEVER explain why a value is null. If unknown, set the field to null and stop.

3. Each attribute schema description (e.g. an "Industry sector" field whose description
   reads "Industry classification, single word where possible") tells you the FORMAT a
   real value should take. The description text is NEVER itself a value. NEVER copy
   schema description text into the field.

4. The literal strings "null", "N/A", "Not specified", "unknown", "none", "not provided",
   or any sentence describing absence are NOT valid values. If no value is supported by
   the source text, set the field to null (or omit it) — do not write a sentence.

5. Each attribute value must be a short, well-formed instance of the type the field
   describes (a phone number, an industry name, a URL, a postal address). If you cannot
   produce a clean value of that type from the source text, the field is null.
"""

attribute_hard_rules_entity = f"""
HARD RULES — violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from text in MESSAGES,
   (b) the existing value already on the ENTITY (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.
{_attribute_hard_rules_common}
6. NEVER infer attribute values from the entity's name, from related entities, from
   generic world knowledge, or from prior summaries. Only verbatim or directly normalized
   text from MESSAGES qualifies as a new value.

7. If MESSAGES contain no information about an attribute, leave the existing entity
   value unchanged. If the entity has no existing value, the field is null.

EXAMPLES

ENTITY: {{"name": "Sam Rivera", "phones": "415-555-0142"}}
MESSAGES contain no phone information for Sam.
GOOD → "phones": "415-555-0142"   (preserved existing value)
BAD  → "phones": "415-555-0142 (implied by original entity, but no new information in
        messages, retaining original value as per instruction...)"

ENTITY: {{"name": "Northwind", "industry": null}}
MESSAGES mention Northwind only as the platform some content was posted to.
GOOD → "industry": null   (no explicit industry classification was stated)
BAD  → "industry": "Content platform, SaaS (implied by usage context, though not stated
        explicitly as industry classification...)"

ENTITY: {{"name": "Priya"}}
MESSAGES contain no phone for Priya, but discuss a project she contributed to.
GOOD → "phones": null
BAD  → "phones": "Worked with Lin and Marco on the Q3 launch..."   (off-topic content dump)
"""

attribute_hard_rules_fact = f"""
HARD RULES — violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from the FACT,
   (b) the existing value already in EXISTING_ATTRIBUTES (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.
{_attribute_hard_rules_common}
6. Use REFERENCE_TIME to resolve any relative temporal expressions in the fact.

7. Preserve existing attribute values unless the FACT explicitly provides a new value.
"""

attribute_hard_rules_summarize_context = f"""
HARD RULES for attribute extraction — violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from text in MESSAGES,
   (b) the existing value already on the ENTITY (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.
{_attribute_hard_rules_common}
6. NEVER infer attribute values from the entity name, from related entities, from
   generic world knowledge, or from prior summaries. Only verbatim or directly normalized
   text from MESSAGES qualifies as a new value.

7. If MESSAGES contain no information about an attribute, leave the existing entity
   value unchanged. If the entity has no existing value, set the field to null.
"""

combined_negative_examples = """
<NEGATIVE EXAMPLES>
Each example shows the source phrasing, what NOT to extract as an entity, and
what to keep instead. The skipped content still survives — inside fact text on
the surviving entity.

A) Multiple-choice / response-option scaffolding
   Source (assistant): "Reply with one of: Strongly disagree, Disagree, Agree,
   or Strongly agree."
   SKIP entities: "Agree", "Strongly disagree", "the four answers", "responses".
   KEEP: nothing — this is template instruction, not a fact about the user.

B) Specific clock times
   Source: "The sun rises around 8:47 am in Stockholm on the winter solstice."
   SKIP entities: "8:47 am", "2:48 pm".
   KEEP: "Stockholm", "winter solstice". The time stays in the fact text:
   "The sun rises around 8:47 am in Stockholm on the winter solstice."

C) Quantities / durations / prices / recipe amounts
   Source: "Berlin and London experience approximately 7.5 hours of daylight
   on the winter solstice."
   SKIP entities: "7.5 hours of daylight", "7-8 hours", "6 hours of daylight".
   KEEP: "Berlin", "London", "winter solstice". Duration stays inside the fact.

   Source: "Mix 1 cup granulated white sugar into 4 cups water to make nectar."
   SKIP entities: "1 cup granulated white sugar", "4 cups water".
   KEEP: "sugar-water nectar" (the recipe topic). The amounts stay in the fact.

D) Geographic coordinates
   Source: "Melbourne is located approximately 37 degrees south of the equator;
   Stockholm is around 59 degrees north."
   SKIP entities: "37 degrees south of the equator", "59 degrees north".
   KEEP: "Melbourne", "Stockholm", "equator". The latitude stays in the fact.

E) Imperative verb-phrase advice from tip lists
   Source (assistant): "Tips for saving money on groceries: Buy in bulk; Cook
   in bulk; Plan your meals; Shop sales; Use cashback apps."
   SKIP entities: "Buy in bulk", "Cook in bulk", "Plan your meals",
   "Shop sales", "Use cashback apps".
   KEEP: "saving money on groceries" (the topical noun phrase). Each tip lives
   inside a fact attached to that topic, not as its own node.

F) Quoted slogans / idioms / loaded phrases
   Source (user): "I believe in 'from each according to his ability, to each
   according to his need' — the rich are too highly taxed though."
   SKIP entities: "from each according to his ability...", "the enemy of my
   enemy", "the rich", "genuinely disadvantaged".
   KEEP: the User entity. The belief and opinion go into facts in plain
   language (e.g. user -> BELIEVES_IN -> Marxist distribution principle).

G) Direct speaker-to-target edges (no fragmenting through scenery)
   Source: Calvin: "I took that pic in Tokyo last night. The skyline was
   stunning! [...]"  Dave: "Wow, the night skyline really pops with those
   city lights. I gotta take a trip there soon!"  Calvin (later): "Touring
   with Frank Ocean last week was wild. Tokyo was unreal — the crowd was
   insane."
   SKIP edge sources/targets: city lights -> Tokyo,
        night skyline -> Tokyo, insane crowd -> Tokyo.
   KEEP edges: Calvin -> TOOK_PHOTO_IN -> Tokyo
               Calvin -> PERFORMED_IN -> Tokyo
               Calvin -> TOURED_WITH -> Frank Ocean
               Dave -> WANTS_TO_VISIT -> Tokyo
   Descriptive scenery ("stunning skyline", "city lights pop", "insane
   crowd") goes inside the fact text on these direct edges, not as its
   own edges.
</NEGATIVE EXAMPLES>
"""
