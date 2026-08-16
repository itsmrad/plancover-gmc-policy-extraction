@AGENTS.md

1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.

If multiple interpretations exist, present them - don't pick silently.

If a simpler approach exists, say so. Push back when warranted.

If something is unclear, stop. Name what's confusing. Ask.

2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.

No abstractions for single-use code.

No "flexibility" or "configurability" that wasn't requested.

No error handling for impossible scenarios.

If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.

Don't refactor things that aren't broken.

Match existing style, even if you'd do it differently.

If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.

Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"

"Fix the bug" → "Write a test that reproduces it, then make it pass"

"Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]


5. Professional Scalable Folder Structure

Use a clean, scalable **feature-first architecture** organized around developer workflow and technical layers.

Keep files easy to identify, maintain, and extend.

* Page-specific components: `components/landing/component-name.tsx`
* Shared components: `components/shared/`
* Reusable UI components: `components/ui/`
* Feature logic: `features/[feature-name]/`
* Global utilities, types, services, config, and libraries in their respective folders

Keep feature-specific components, hooks, services, types, and tests colocated. Avoid vague folders and unnecessary global files. Overall make the use professional folder structure that is highly maintainable and is easy to navigate and understand.


# Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

Some more points :

every new feature is linked to a branch like a new xyz branch which should definitely be branched off from main unless specified, you will make changes in the new branch and commit on that branch, there can be multiple commits on one branch, and after completion push it to github then i will manually merge it

make sure to reuse code as much as possible 

make sure to follow professional developer standards and make sure to include best developer practices

the prompt given below is for only instruction, if you believe there is a better way to handle this specific problem ask me and then go forward, make sure to think what must be optimal before execution

do not get involved in making it too optimal that you forget to execute it and keep thinking, i need a mid ok ok optimal level that works not something state of the art. value execution before too much optimization

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.