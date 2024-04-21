import re
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


def preprocess_paragraph(paragraph):
    cleaned_paragraph = re.sub(r"[^\w\s]", "", paragraph)
    cleaned_paragraph = cleaned_paragraph.lower()
    words = cleaned_paragraph.split()
    return words


def build_graph(words):
    graph = defaultdict(list)
    for i in range(len(words) - 1):
        graph[words[i]].append(words[i + 1])
    return graph


def visualize_graph(graph):
    G = nx.DiGraph()
    for node, edges in graph.items():
        for edge in edges:
            G.add_edge(node, edge)
    nx.draw(G, with_labels=True)
    plt.show()


paragraph = """Cricket is the most popular sport in Pakistan. Almost all districts and neighbourhoods in Pakistan have a cricket team and people start playing from a young age. Pakistan has won international cricket events, which include the 1992 Cricket World Cup, the 2009 ICC World Twenty20 and the 2017 ICC Champions Trophy besides finishing as runner-up in the 1999 Cricket World Cup, 2007 ICC World Twenty20 and the 2022 T20 World Cup. Pakistan also won the ACC Asia Cup in 2000 and 2012 and all three versions of the Austral-Asia Cup.
Pakistan's cricket teams take part in domestic competitions such as the Quaid-e-Azam Trophy, the Patron's Trophy, ABN-AMRO Twenty-20 Cup, and the ABN-AMRO Champions Trophy. Pakistan Cricket Board also organize a franchise based T20 cricket league known as the Pakistan Super League.
International Test and one-day matches are played between the Pakistan national cricket team and foreign opponents regularly. Women's cricket is also very popular, with Kiran Baluch holding the current record for the highest score in a women's test match with her innings of 242. The Pakistan Cricket Board controls both the men's and women's games. The 2020 Pakistan Super League events was hosted entirely by Pakistan.
Notable cricketers from Pakistan include Aaqib Javed, Ramiz Raja, Babar Azam, Shoaib Akhtar, Younis Khan, Saqlain Mushtaq, Mushtaq Ahmed, Abdul Qadir, Wasim Akram, Zaheer Abbas, Javed Miandad, Saeed Anwar, Muhammad Yousaf, Inzamam-ul-Haq, Waqar Younis, Shahid Afridi, the Mohammad brothers (Hanif, Mushtaq, Sadiq and Wazir) and Imran Khan. Imran Khan has been named in the ICC Cricket Hall of Fame. Saeed Anwar's 194 runs against India remained the record for most runs by a batsman in an ODI for 11 years which was broken by Fakhar Zaman's 210 against Zimbabwe in 2018. Shoaib Akhtar holds the record of delivering the fastest delivery in the history of cricket. Shahid Afridi holds numerous records i.e. the 3rd fastest century in ODIs, and the highest number of sixes in international cricket. Wasim Akram at the time of his retirement had taken the most wickets in ODIs. Muhammad Yousuf has scored the most Test runs in a calendar year. The structure of domestic cricket in Pakistan at the highest level has changed many times since 1947 with the latest restructure being enforced in 2019.
"""

words = preprocess_paragraph(paragraph)
print(words)

graph = build_graph(words)

visualize_graph(graph)
