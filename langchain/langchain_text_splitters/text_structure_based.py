from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
FC Barcelona
 Founded in 1899, FC Barcelona is one of the most popular and successful football clubs in the
 world. Known for its attacking style of play and youth academy La Masia, it has produced legends
 like Lionel Messi, Xavi, and Iniesta. The club plays its home games at Camp Nou and boasts
 numerous La Liga and Champions League titles.
 Real Madrid CF
 Real Madrid, founded in 1902, is a symbol of football excellence. It has won more UEFA Champions
 League titles than any other club. With stars like Cristiano Ronaldo and Zinedine Zidane having
 graced the team, Real Madrid plays at the Santiago Bernabeu Stadium and has a rich legacy of
 domestic and international success.
 Manchester United
 Manchester United is a historic English football club founded in 1878. Based at Old Trafford, it has a
 huge global following. The club has won numerous Premier League titles and Champions League
 trophies. Legendary players like George Best, Eric Cantona, and Wayne Rooney have contributed
 to its rich heritage.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)
print(chunks[0])