Here is another Python/Flask/javascript example of a wild GPT API use case.  First throw any documents you want into a Documents folder.  Get an OpenAI API key and store it as APIkey.txt in the base folder.  Put index.html in a subdirectory "Templates". Run Embeddings.py to take all of your documents, chunk them up using a rolling window, and embed each chunk into high-dimensional space.  Then run app.py as a local Flask app, and you will be able whatever questions you want of your own research papers!

A few caveats on what this will do a poor job with here: https://twitter.com/Afinetheorem/status/1634516697515261953

On the other hand, the code as written is not even in the ballpark of optimized, either in terms of how it chunks text, or in terms of how it pre-processes queries.  This is a "it's good enough" solution, not a perfect one.

Have fun building off of it!
