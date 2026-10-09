## Missing information

The request is written in the client's own words. It will not spell out every
detail you need — data conventions, units, edge-case handling and similar points
are often left implicit because the client considers them obvious.

The client is reachable. Before writing any code, ask the client about anything
you need settled, and keep asking until you have what you need. Do not start
the work until then.

Ask about two kinds of things:

- Things you are unsure about — the request and the data do not settle them.
- Things you believe you already understand from the data. State your reading
  as the default option and let the client confirm or correct it. Having seen
  something in the data is not the same as having understood it correctly.

You may ask across several rounds.

When you ask, put the questions at the end of your reply under a heading
`## Questions for the client`, one question per line, and stop there — the client
will answer in the next message. (Asking in plain prose also works; the heading
just makes the exchange cleaner.)

Proceeding on an unconfirmed reading counts as a failure even if the guess
turns out right. Record in `NOTES.md` any assumption the client did not settle.
