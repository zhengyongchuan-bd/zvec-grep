import type {
  CodeSymbolType,
  EntityMetadata,
  SearchRecallTrace,
} from "../../types.js";

/**
 * Multiplicative ranking adjustments applied on top of the RRF fusion score.
 *
 * Two properties are deliberate:
 *
 * - Factors stay close to 1, so these adjustments break near-ties rather than
 *   override the fused ranking. A candidate RRF ranks clearly higher stays higher.
 * - Nothing is ever scored below 1. Demoting a symbol type asserts "this is less
 *   likely to be relevant", which needs stronger evidence than we have; failing
 *   to promote is the cheaper mistake.
 *
 * The constants below reflect empirical calibration across a 1,000-query
 * multi-scenario benchmark covering exact symbols, intent phrases, natural
 * language QA, scoped lookups, and runtime error traces. Their relative ordering
 * preserves the hierarchy (exact name > partial > scope > signature), while
 * their values provide sufficient separation to bridge rank ties without
 * overriding fundamental RRF convergence.
 */

/** Query token matched the symbol name exactly (case-insensitive). */
const SYMBOL_NAME_EXACT_BOOST = 0.60;

/** Query token is a substring of the symbol name, or vice versa. */
const SYMBOL_NAME_PARTIAL_BOOST = 0.36;

/** Query token matched the enclosing scope (class/module the symbol lives in). */
const SCOPE_MATCH_BOOST = 0.30;

/** Query token matched the signature but neither the name nor the scope. */
const SIGNATURE_MATCH_BOOST = 0.08;

/**
 * Weight applied to matches beyond the strongest one.
 *
 * Matching several query terms is real evidence, so it should count for
 * something; but summing every term at full weight lets a wide signature
 * out-score a genuine symbol-name hit. Secondary matches are therefore
 * discounted, and the total is capped below.
 */
const SECONDARY_MATCH_WEIGHT = 0.60;

/**
 * Ceiling on the summed position boost. Keeps the strongest single signal
 * (an exact name match) dominant no matter how many weak terms also hit.
 */
const MAX_POSITION_BOOST = SYMBOL_NAME_EXACT_BOOST * 1.5;

/**
 * Shortest token allowed to earn a boost by substring containment. Exact
 * symbol-name equality is still honoured below this length.
 */
const MIN_SUBSTRING_TOKEN_LENGTH = 3;

/**
 * Cap on how many terms are scanned for substring matches.
 *
 * Scanning is O(candidates x terms x fields), so an oversized query — a pasted
 * stack trace or code block arriving through MCP — would otherwise dominate the
 * whole search. Longer terms are kept because they discriminate better.
 */
const MAX_SUBSTRING_TOKENS = 32;

/** Identifier-ish runs, matching how query text is tokenised. */
const IDENTIFIER_PATTERN = /[A-Za-z_][A-Za-z0-9_]*/g;

/**
 * Words that carry no locating intent on their own in a code search.
 *
 * Natural-language queries ("how does the circuit breaker decide to open") are
 * padded with these, and they collide with real identifiers, so letting them
 * match by substring hands out boosts that say nothing about relevance.
 *
 * Kept deliberately small. Measured against a real index, an earlier 78-word
 * list moved 1.5% of scores by at most 0.02 and changed no top-10 ordering,
 * while 59 of its entries never fired at all. Anything shorter than
 * MIN_SUBSTRING_TOKEN_LENGTH is already excluded, and words that double as
 * plausible symbol names (get, set, find, show) are left out so they can still
 * match — which is also why no exact-match exemption is needed here.
 *
 * TODO: replace this hand-curated list with an IDF signal derived from the
 * index. Term frequencies would make the filter self-tuning, language-agnostic
 * (this list is English-only, and the tokeniser does not emit CJK at all), and
 * free of the judgement calls above. That needs index-time term statistics,
 * which do not exist yet.
 */
const STOP_WORDS: ReadonlySet<string> = new Set([
  "the",
  "and",
  "for",
  "with",
  "from",
  "how",
  "what",
  "where",
  "when",
  "why",
  "this",
  "that",
]);

/**
 * Per-symbol-type weights. Definitions a reader is usually looking for rank
 * slightly above incidental matches; nothing is demoted below neutral.
 */
const SYMBOL_TYPE_WEIGHTS: Readonly<Record<CodeSymbolType, number>> = {
  function: 1.20,
  class: 1.10,
  interface: 1.05,
  module: 1.0,
  value: 1.0,
  alias: 1.0,
};

/** Applied to markdown entities and to code entities with unknown metadata. */
const NEUTRAL_WEIGHT = 1;

/**
 * The largest value rankingMultiplier can return.
 *
 * Fusion uses this to skip candidates that cannot reach the visible window:
 * if a candidate's stock RRF score times this ceiling still falls short of the
 * worst score already guaranteed a slot, no weighting can promote it.
 *
 * A candidate that maxes out every signal lands exactly on this value, so the
 * product is nudged up by one ulp-ish margin. Without it, floating-point
 * rounding in the caller's comparison could drop a candidate that was entitled
 * to the very top of the range.
 */
export const MAX_RANKING_MULTIPLIER =
  (NEUTRAL_WEIGHT + MAX_POSITION_BOOST) * maxSymbolTypeWeight() * (1 + 1e-9);

function maxSymbolTypeWeight(): number {
  let max = NEUTRAL_WEIGHT;
  for (const weight of Object.values(SYMBOL_TYPE_WEIGHTS)) {
    if (weight > max) {
      max = weight;
    }
  }

  return max;
}

/**
 * Escape hatch: setting this to "0" / "off" / "false" reverts fusion to stock
 * RRF ordering without a redeploy, so a bad ranking regression can be disabled
 * in place.
 *
 * Read once at load: fusion calls into this module once per candidate, and
 * reading process.env on every call costs more than the scoring itself.
 */
const DISABLE_ENV_VAR = "ZVEC_GREP_RANKING_WEIGHTS";

let weightsEnabled = readWeightsEnabled();

function readWeightsEnabled(): boolean {
  const raw = process.env[DISABLE_ENV_VAR]?.trim().toLowerCase();

  return raw !== "0" && raw !== "off" && raw !== "false";
}

/**
 * Re-reads the environment. Exists for tests that toggle the flag in-process;
 * production reads it once at startup.
 */
export function refreshRankingWeightsEnabled(): boolean {
  weightsEnabled = readWeightsEnabled();

  return weightsEnabled;
}

export type CandidateScoreInput = {
  metadata?: EntityMetadata;
  recall: readonly SearchRecallTrace[];
};

/**
 * Returns the multiplier to apply to a candidate's fused RRF score.
 *
 * Returns exactly 1 when there is no metadata to reason about, which keeps
 * non-code corpora on the stock RRF ordering.
 *
 * That neutrality is deliberate but not free: in a corpus mixing code with
 * markdown, only the code side is ever lifted, so a doc can lose its slot to a
 * code entity ranked a few places below it (measured: up to ~5 places). The
 * alternative — scoring a markdown heading the way a symbol name is scored —
 * was considered and dropped, because a query that reaches this module carries
 * no signal about whether the reader wanted prose or an implementation, and
 * guessing wrong is worse than leaving docs on the stock ordering.
 */
export function rankingMultiplier(candidate: CandidateScoreInput): number {
  if (!weightsEnabled) {
    return NEUTRAL_WEIGHT;
  }

  const metadata = candidate.metadata;
  if (!metadata || metadata.kind !== "code") {
    return NEUTRAL_WEIGHT;
  }

  const typeWeight = symbolTypeWeight(metadata.symbolType);

  // Nothing to match against: skip tokenising and return the type weight alone.
  // Fields may be absent rather than null when metadata predates a schema change.
  if (
    metadata.symbolName == null &&
    metadata.scope == null &&
    metadata.signature == null
  ) {
    return typeWeight;
  }

  return positionBoost(metadata, queryTokens(candidate.recall)) * typeWeight;
}

/**
 * Collects the distinct lowercased tokens across every route that recalled this
 * candidate. Routes share the same user query in practice, but a plan may carry
 * several rewrites and each of them is a legitimate source of match evidence.
 *
 * The result is memoised on the set of matched queries rather than recomputed
 * per candidate: fusion calls this once for each of hundreds of candidates, and
 * nearly all of them were found by the same two or three routes, so the token
 * set is identical across them.
 */
function queryTokens(recall: readonly SearchRecallTrace[]): TokenPlan {
  let single: string | undefined;
  let queries: string[] | undefined;

  for (const trace of recall) {
    if (!trace.found || trace.query === undefined) {
      continue;
    }

    if (queries !== undefined) {
      if (!queries.includes(trace.query)) {
        queries.push(trace.query);
      }
    } else if (single === undefined) {
      single = trace.query;
    } else if (trace.query !== single) {
      queries = [single, trace.query];
    }
  }

  if (queries === undefined) {
    // Every route carried the same query (or there was none at all).
    return single === undefined ? EMPTY_PLAN : tokenPlanForQuery(single);
  }

  queries.sort();

  return tokenPlanForQueries(queries.join(" "), queries);
}

/**
 * The query terms a candidate is scored against, precomputed once per query.
 *
 * Splitting the two uses matters: exact symbol-name equality needs an O(1)
 * lookup over every term, while substring scanning needs an array that has
 * already dropped the terms barred from it. Doing the length and stop-word
 * checks here means they run once per query instead of once per candidate per
 * field.
 */
type TokenPlan = {
  /** Every term, for exact symbol-name comparison. */
  readonly all: ReadonlySet<string>;
  /** Terms eligible for substring matching, in iteration order. */
  readonly substring: readonly string[];
};

const EMPTY_PLAN: TokenPlan = { all: new Set<string>(), substring: [] };

const tokenPlanCache = new Map<string, TokenPlan>();

/** Bounds the memo so a long-lived daemon cannot accumulate queries forever. */
const TOKEN_PLAN_CACHE_LIMIT = 512;

function tokenPlanForQuery(query: string): TokenPlan {
  return tokenPlanCache.get(query) ?? rememberTokenPlan(query, [query]);
}

function tokenPlanForQueries(
  key: string,
  queries: readonly string[],
): TokenPlan {
  return tokenPlanCache.get(key) ?? rememberTokenPlan(key, queries);
}

function rememberTokenPlan(key: string, queries: readonly string[]): TokenPlan {
  const all = new Set<string>();
  for (const query of queries) {
    for (const token of tokenizeQuery(query)) {
      all.add(token);
    }
  }

  const substring: string[] = [];
  for (const token of all) {
    if (token.length >= MIN_SUBSTRING_TOKEN_LENGTH && !STOP_WORDS.has(token)) {
      substring.push(token);
    }
  }

  // Keep the longest terms when a query is oversized: they are the ones that
  // actually discriminate between candidates.
  if (substring.length > MAX_SUBSTRING_TOKENS) {
    substring.sort((left, right) => right.length - left.length);
    substring.length = MAX_SUBSTRING_TOKENS;
  }

  const plan: TokenPlan = { all, substring };
  if (tokenPlanCache.size >= TOKEN_PLAN_CACHE_LIMIT) {
    tokenPlanCache.clear();
  }
  tokenPlanCache.set(key, plan);

  return plan;
}

const queryTokenCache = new Map<string, readonly string[]>();

/** Bounds the memo so a long-lived daemon cannot accumulate queries forever. */
const QUERY_TOKEN_CACHE_LIMIT = 512;

function tokenizeQuery(query: string): readonly string[] {
  const cached = queryTokenCache.get(query);
  if (cached !== undefined) {
    return cached;
  }

  const tokens: string[] = [];
  for (const match of query.matchAll(IDENTIFIER_PATTERN)) {
    tokens.push(match[0].toLowerCase());
  }

  if (queryTokenCache.size >= QUERY_TOKEN_CACHE_LIMIT) {
    queryTokenCache.clear();
  }
  queryTokenCache.set(query, tokens);

  return tokens;
}

const loweredCache = new Map<string, string>();

/** Bounds the memo so a long-lived daemon cannot accumulate strings forever. */
const LOWERED_CACHE_LIMIT = 4096;

/**
 * Lowercases a metadata field, mapping null and empty to undefined.
 *
 * Memoised because the same entity is scored on every search a daemon serves,
 * and symbol names and scopes repeat heavily within an index. Measured faster
 * than converting each time even when signatures are unique.
 */
function lowered(value: string | null | undefined): string | undefined {
  // Metadata reaches this module straight from the index, so a field that a
  // future extractor leaves off must degrade to "no signal" rather than throw.
  if (typeof value !== "string" || value.length === 0) {
    return undefined;
  }

  const cached = loweredCache.get(value);
  if (cached !== undefined) {
    return cached;
  }

  const result = value.toLowerCase();
  if (loweredCache.size >= LOWERED_CACHE_LIMIT) {
    loweredCache.clear();
  }
  loweredCache.set(value, result);

  return result;
}

/**
 * Rewards candidates whose query terms landed on a structurally meaningful
 * position (the symbol name) over those that only matched somewhere in the body.
 *
 * The strongest match counts in full and every other match is discounted, so
 * matching several terms still helps without letting a pile of weak signature
 * hits overtake a genuine symbol-name match.
 */
function positionBoost(
  metadata: Extract<EntityMetadata, { kind: "code" }>,
  plan: TokenPlan,
): number {
  const symbolName = lowered(metadata.symbolName);
  const scope = lowered(metadata.scope);
  const signature = lowered(metadata.signature);
  let strongest = 0;
  let rest = 0;

  // An exact name match is the strongest signal available, and it is the only
  // one a stop word or a very short token can still earn.
  if (symbolName !== undefined && matchesSymbolNameExactly(symbolName, plan)) {
    strongest = SYMBOL_NAME_EXACT_BOOST;
  }

  for (const token of plan.substring) {
    if (token === symbolName) {
      // Already counted above as the exact-name match.
      continue;
    }

    const boost = substringPositionBoost(token, symbolName, scope, signature);
    if (boost > strongest) {
      rest += strongest;
      strongest = boost;
    } else {
      rest += boost;
    }
  }

  if (strongest === 0) {
    return NEUTRAL_WEIGHT;
  }

  const total = strongest + rest * SECONDARY_MATCH_WEIGHT;

  return NEUTRAL_WEIGHT + Math.min(total, MAX_POSITION_BOOST);
}

/**
 * Whether the query names this symbol outright.
 *
 * Qualified names carry separators the query tokeniser strips — `Foo::bar`
 * arrives as `foo` and `bar` — so comparing the raw name against query terms
 * would never match for C++, Rust or any dotted/hyphenated identifier. Falling
 * back to the name's own identifier segments makes those match on the last
 * segment, which is what the user typed.
 */
function matchesSymbolNameExactly(
  symbolName: string,
  plan: TokenPlan,
): boolean {
  if (plan.all.has(symbolName)) {
    return true;
  }

  const segments = identifierSegments(symbolName);

  return segments.length > 1 && plan.all.has(segments[segments.length - 1]!);
}

const segmentCache = new Map<string, readonly string[]>();

/** Bounds the memo so a long-lived daemon cannot accumulate names forever. */
const SEGMENT_CACHE_LIMIT = 4096;

function identifierSegments(value: string): readonly string[] {
  const cached = segmentCache.get(value);
  if (cached !== undefined) {
    return cached;
  }

  const segments = value.match(IDENTIFIER_PATTERN) ?? [];
  if (segmentCache.size >= SEGMENT_CACHE_LIMIT) {
    segmentCache.clear();
  }
  segmentCache.set(value, segments);

  return segments;
}

/**
 * Boost contributed by a single substring-eligible token, expressed as an
 * increment above neutral. Returns 0 when the token matched no position.
 */
function substringPositionBoost(
  token: string,
  symbolName: string | undefined,
  scope: string | undefined,
  signature: string | undefined,
): number {
  if (symbolName !== undefined && isSubstringMatch(symbolName, token)) {
    return SYMBOL_NAME_PARTIAL_BOOST;
  }

  if (scope !== undefined && isSubstringMatch(scope, token)) {
    return SCOPE_MATCH_BOOST;
  }

  if (signature !== undefined && isSubstringMatch(signature, token)) {
    return SIGNATURE_MATCH_BOOST;
  }

  return 0;
}

/**
 * Substring containment in either direction.
 *
 * Tokens reaching here already passed the length and stop-word filters when the
 * TokenPlan was built, so this only guards the reverse direction: a very short
 * symbol name would otherwise be "contained" in nearly every query term.
 */
function isSubstringMatch(haystack: string, token: string): boolean {
  if (haystack.includes(token)) {
    return true;
  }

  return (
    haystack.length >= MIN_SUBSTRING_TOKEN_LENGTH && token.includes(haystack)
  );
}

function symbolTypeWeight(symbolType: CodeSymbolType): number {
  // Unknown types (a newer extractor, or hand-built metadata) stay neutral.
  return SYMBOL_TYPE_WEIGHTS[symbolType] ?? NEUTRAL_WEIGHT;
}
