/**
 * Cross-module state, in one place.

`currentInference` and the transcript are read by more than one module. A module
that reached into another's top-level `let` would make the import graph a lie,
so the shared values live here and everything else imports them.
 */

/** Most recent /api/infer or /api/chat hierarchy payload. */
export const store = {
    currentInference: null,
    staticMatrices: null,
    chatHistory: [],
    chatBusy: false,
};
