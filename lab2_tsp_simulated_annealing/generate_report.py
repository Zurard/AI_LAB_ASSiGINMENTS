# Build the Lab Assignment 2 Word report from a live SA run
# so the numbers, tables and figures match the code.

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from cities import CITIES
from tsp_sa import format_tour, run_experiment


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Lab_Assignment_2_TSP_Simulated_Annealing.docx")


# ---------- tiny style helpers ----------

def set_run(run, name="Calibri", size=12, bold=False, italic=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def shade_cell(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_cell_border(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "8FA4B8")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def para(doc, text, *, size=12, bold=False, italic=False, center=False,
         space_after=8, space_before=0, first_line=True, color=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line:
        p.paragraph_format.first_line_indent = Cm(0.75)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    set_run(run, size=size, bold=bold, italic=italic, color=color)
    return p


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    return h


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.first_line_indent = Cm(0)
    run = p.add_run(text)
    set_run(run, size=10, italic=True, color=(80, 80, 80))
    return p


def add_picture(doc, path, width=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(path, width=Inches(width))
    return p


def add_code(doc, code, title=None):
    if title:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.first_line_indent = Cm(0)
        run = p.add_run(title)
        set_run(run, size=11, bold=True, italic=True, color=(31, 58, 95))

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    shade_cell(cell, "F4F6F8")
    set_cell_border(cell)
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(code.rstrip() + "\n")
    set_run(run, name="Consolas", size=8.5, color=(30, 30, 30))
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def fill_header_row(row, texts):
    for i, text in enumerate(texts):
        cell = row.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        set_run(run, size=10, bold=True, color=(255, 255, 255))
        shade_cell(cell, "1F3A5F")
        set_cell_border(cell)


def fill_row(row, texts, center=True, shade=None):
    for i, text in enumerate(texts):
        cell = row.cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        if center:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(str(text))
        set_run(run, size=10)
        if shade:
            shade_cell(cell, shade)
        set_cell_border(cell)


def extract(path, start, end):
    lines = open(path, encoding="utf-8").read().splitlines()
    chunk = lines[start - 1:end]
    # drop leading extra blanks
    while chunk and chunk[0].strip() == "":
        chunk.pop(0)
    return "\n".join(chunk)


# ---------- document ----------

def build():
    print("Running Simulated Annealing and drawing figures...")
    data = run_experiment(fig_dir=os.path.join(HERE, "figures"))
    fig = data["fig_dir"]
    sa = data["sa"]
    extra = data["extra"]
    sa_nn = data["sa_from_nn"]

    improve = 100.0 * (data["init_cost"] - sa["cost"]) / data["init_cost"]
    vs_nn = 100.0 * (data["nn_cost"] - sa["cost"]) / data["nn_cost"]

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(12)

    # ===== TITLE =====
    banner = doc.add_paragraph()
    banner.alignment = WD_ALIGN_PARAGRAPH.CENTER
    banner.paragraph_format.space_after = Pt(4)
    r = banner.add_run("ARTIFICIAL INTELLIGENCE LABORATORY")
    set_run(r, size=13, bold=True, color=(31, 58, 95))

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(6)
    r = title.add_run(
        'Lab Assignment 2 : "Solving the Traveling Salesman Problem '
        'for a Rajasthan Tour using Simulated Annealing"'
    )
    set_run(r, size=16, bold=True, color=(31, 58, 95))

    meta = [
        ("Name", "Shreetej Meshram"),
        ("ID", "202352333"),
        ("Problem", "Traveling Salesman Problem (TSP)"),
        ("Method", "Simulated Annealing"),
        ("Instance", "22 tourist locations in Rajasthan"),
        ("Language", "Python 3"),
    ]
    t = doc.add_table(rows=len(meta), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (k, v) in enumerate(meta):
        fill_row(t.rows[i], [k, v], center=False, shade="F4F6F8" if i % 2 == 0 else None)
        t.rows[i].cells[0].paragraphs[0].runs[0].bold = True

    para(doc, "", first_line=False, space_after=4)

    # ===== 1 AIM =====
    heading(doc, "1. Aim", 1)
    para(doc,
         "The aim of this assignment is to plan a low-cost cyclic tour of "
         "Rajasthan for visiting relatives by treating the problem as an "
         "instance of the Traveling Salesman Problem (TSP) and solving it "
         "with Simulated Annealing. At least twenty important tourist "
         "locations are chosen. The cost of travelling between any pair of "
         "cities is taken to be proportional to the geographic distance "
         "between them. The program must return a cycle that visits every "
         "chosen city exactly once and returns to the starting city, with "
         "the total distance kept as small as possible.")
    para(doc,
         "The supporting objectives are: (i) to formulate the Rajasthan "
         "tour as a complete, symmetric TSP; (ii) to implement Simulated "
         "Annealing with a 2-opt neighbourhood and the Metropolis "
         "acceptance rule; (iii) to compare the annealed tour with a random "
         "start and with a greedy nearest-neighbour construction; and "
         "(iv) to study how the tour cost falls as the temperature is cooled.")

    # ===== 2 QUESTION DISCUSSION =====
    heading(doc, "2. Question Discussion", 1)

    heading(doc, "2.1 What the question asks", 2)
    para(doc,
         "The Traveling Salesman Problem is easy to state and hard to solve. "
         "We are given a graph whose nodes are cities and whose edges are "
         "labelled with the cost of travelling between those cities. The "
         "task is to find a cycle that contains every city exactly once "
         "and whose total cost is as small as possible. For this lab the "
         "graph is built from the state of Rajasthan. Relatives are assumed "
         "to arrive next week, so a single closed tour that starts and ends "
         "in Jaipur (the usual arrival city) is required. The question also "
         "tells us that it is reasonable to take travel cost as proportional "
         "to distance. That one modelling choice turns a vague tourism "
         "question into a well-defined optimisation problem.")

    heading(doc, "2.2 Why an exact search is not practical", 2)
    para(doc,
         "A tour is a permutation of the n cities. Because the tour is a "
         "cycle, rotations of the same sequence are the same tour, and the "
         "opposite direction is the same tour as well. The number of "
         "distinct tours is therefore (n - 1)! / 2. For n = 22 this is")
    para(doc, "21! / 2  =  2.56 x 10^19", center=True, first_line=False,
         bold=True, space_after=10)
    para(doc,
         "Even if a computer could score one billion tours every second, "
         "listing them all would take hundreds of years. Dynamic programming "
         "(Held-Karp) reduces the work to roughly n * 2^n states, which is "
         "still about 92 million states for 22 cities and is heavy in plain "
         "Python. The assignment therefore asks for Simulated Annealing: a "
         "local-search method that does not promise the true optimum, but "
         "can reach a very good tour in a few seconds.")

    heading(doc, "2.3 Choosing the twenty-two locations", 2)
    para(doc,
         "Rajasthan is large and the tourist map is not a straight line. "
         "The twenty-two places below cover the main circuits that visitors "
         "actually use: the Jaipur-Ajmer-Pushkar belt, the desert cities of "
         "Jodhpur, Jaisalmer, Bikaner, Osian, Nagaur and Barmer, the lake "
         "and fort belt of Udaipur, Kumbhalgarh, Ranakpur, Nathdwara, "
         "Chittorgarh, Mount Abu and Dungarpur, and the eastern wildlife "
         "and palace towns of Ranthambore, Bundi, Kota, Alwar, Bharatpur "
         "and Mandawa. Each place is represented by one latitude-longitude "
         "pair near the city centre. Two nearby towns (Ajmer and Pushkar) "
         "are kept as separate stops because they are distinct visits even "
         "though the road between them is short.")

    heading(doc, "2.4 Cost model", 2)
    para(doc,
         "Road distances, bus fares and hotel nights are not published as a "
         "clean matrix for every pair of these towns. The assignment allows "
         "us to treat cost as proportional to distance, so a single constant "
         "of proportionality can be ignored. The program therefore minimises "
         "kilometres. The distance between two GPS points is the great-circle "
         "(haversine) distance on a sphere of radius 6371 km. The resulting "
         "graph is complete and symmetric: there is an edge between every "
         "pair of cities, and d(i, j) = d(j, i). A tour is a Hamiltonian "
         "cycle. Its cost is the sum of the twenty-two edge lengths, "
         "including the last hop back to Jaipur.")
    para(doc,
         "This is an approximation of real travel. Haversine distance is "
         "shorter than the road network, and it ignores one-way streets, "
         "hills and the fact that some desert roads are slower than the "
         "Jaipur-Ajmer highway. For a week-scale planning exercise the "
         "ranking of tours is still useful: a tour that is hundreds of "
         "kilometres shorter on the map is almost always cheaper and faster "
         "on the ground as well.")

    heading(doc, "2.5 Simulated Annealing, in the language of this problem", 2)
    para(doc,
         "Simulated Annealing copies the way a metal is cooled so that its "
         "atoms settle into a low-energy crystal. In TSP the 'energy' of a "
         "state is the length of the current tour. The algorithm keeps one "
         "current tour. At every step it proposes a nearby tour by reversing "
         "a random segment (a 2-opt move). If the neighbour is shorter, it "
         "is accepted at once. If the neighbour is longer by an amount "
         "delta, it is still accepted with probability")
    para(doc, "P(accept)  =  exp( - delta / T )", center=True,
         first_line=False, bold=True, space_after=10)
    para(doc,
         "where T is the current temperature. Early on, T is large, so even "
         "clearly worse tours are often accepted. That is how the search "
         "escapes a local minimum: it is allowed to cross a longer stretch "
         "of desert in order to untangle a pair of crossing edges later. "
         "After a fixed number of trials the temperature is multiplied by a "
         "cooling factor alpha (here 0.995). As T falls, the exponential "
         "becomes tiny unless delta is almost zero, and the search behaves "
         "more and more like ordinary hill-climbing. When T is smaller than "
         "a cut-off the run stops and the shortest tour ever seen is returned.")
    para(doc,
         "Two other methods are used only for comparison. A purely random "
         "permutation is the starting state. A nearest-neighbour tour, built "
         "by always jumping to the closest unused city, is a fast greedy "
         "baseline. Simulated Annealing is also restarted from that greedy "
         "tour to check whether a better construction helps the annealer.")

    heading(doc, "2.6 Parameters used in this run", 2)
    para(doc,
         "The numbers below were chosen so that the search is long enough "
         "to improve a 22-city tour, but short enough to finish in a few "
         "seconds on an ordinary laptop. The same seed is reported so that "
         "the tour in this report can be reproduced by running tsp_sa.py.",
         first_line=True)

    ptab = doc.add_table(rows=6, cols=2)
    ptab.alignment = WD_TABLE_ALIGNMENT.CENTER
    fill_header_row(ptab.rows[0], ["Parameter", "Value"])
    fill_row(ptab.rows[1], ["Initial temperature T0", "%.0f" % sa["t0"]])
    fill_row(ptab.rows[2], ["Final temperature Tmin", str(sa["t_min"])])
    fill_row(ptab.rows[3], ["Cooling factor alpha", str(sa["alpha"])])
    fill_row(ptab.rows[4], ["Moves per temperature (inner)", str(sa["inner"])])
    fill_row(ptab.rows[5], ["Random seed (main run)", str(sa["seed"])])
    caption(doc, "Table 1. Simulated Annealing parameters used for the main experiment.")

    # ===== 3 CODE =====
    heading(doc, "3. Code", 1)
    para(doc,
         "The program is split into two files. cities.py stores the twenty-two "
         "places and their coordinates. tsp_sa.py builds the distance matrix, "
         "runs Simulated Annealing, draws the figures and prints the tour. "
         "The snippets below are taken from those files. After each snippet "
         "the same idea is explained in words, so the listing itself is the "
         "walk-through of the algorithm.")

    heading(doc, "3.1 The city list", 2)
    add_code(doc, extract(os.path.join(HERE, "cities.py"), 5, 29),
             "Snippet 1  -  cities.py  (dataset)")
    para(doc,
         "Each row is one stop on the relatives' tour. The first field is "
         "the display name. The next two fields are latitude and longitude "
         "in decimal degrees. Jaipur is stored first and is treated as the "
         "home city when a cycle is printed, but the optimiser itself does "
         "not privilege any index: a cycle can be rotated without changing "
         "its cost. Keeping the data in a plain list (instead of a GIS file) "
         "makes the assignment easy to read and to mark.")

    heading(doc, "3.2 Distance as cost", 2)
    add_code(doc, extract(os.path.join(HERE, "tsp_sa.py"), 22, 41),
             "Snippet 2  -  haversine distance and the cost matrix")
    para(doc,
         "haversine() is the standard formula for the shortest distance over "
         "the earth's surface. The differences of latitude and longitude are "
         "converted to radians. The quantity a is the square of half the "
         "chord length; two times the arcsine of its square root is the "
         "central angle. Multiplying by the earth radius gives kilometres. "
         "build_distance_matrix() fills an n by n table so that every later "
         "cost query is an array look-up. The matrix is symmetric and the "
         "diagonal is zero, which matches a simple undirected TSP.")

    heading(doc, "3.3 Scoring a tour", 2)
    add_code(doc, extract(os.path.join(HERE, "tsp_sa.py"), 46, 52),
             "Snippet 3  -  closed-cycle cost")
    para(doc,
         "A state of the search is just a list of city indices, for example "
         "[0, 2, 1, ...]. tour_cost() walks that list once and adds the "
         "matrix entry from each city to the next. The wrap-around "
         "(i + 1) % n is the return flight or the last road back to the "
         "starting city. Without that last edge the problem would be a "
         "path, not a tour, and the relatives would be left in the last "
         "town. Because every edge is read from the same matrix, any "
         "improvement in this number is a genuine shortening of the cycle.")

    heading(doc, "3.4 The 2-opt neighbour", 2)
    add_code(doc, extract(os.path.join(HERE, "tsp_sa.py"), 61, 70),
             "Snippet 4  -  generating a neighbour by reversing a segment")
    para(doc,
         "Local search needs a neighbourhood: a cheap way to change the "
         "current tour into a similar tour. two_opt_neighbor() picks two "
         "random positions and reverses the cities that lie between them. "
         "In the language of TSP this is a 2-opt move. It deletes two "
         "edges of the cycle and reconnects the four endpoints the other "
         "way. If those two edges were crossing on the map, the reversal "
         "uncrosses them and the length usually drops. If they were not "
         "crossing, the new tour may be longer. That longer neighbour is "
         "exactly the kind of 'uphill' move Simulated Annealing must "
         "sometimes accept. The function copies the list first, so the "
         "current tour is never overwritten unless the move is accepted.")

    heading(doc, "3.5 The annealing loop", 2)
    add_code(doc, extract(os.path.join(HERE, "tsp_sa.py"), 96, 159),
             "Snippet 5  -  Simulated Annealing (core algorithm)")
    para(doc,
         "This is the heart of the assignment. The function is given the "
         "distance matrix and a handful of cooling parameters. A random "
         "permutation (or a caller-supplied start) becomes the current "
         "tour. best remembers the shortest tour seen at any time, which "
         "matters because the search is allowed to get worse.")
    para(doc,
         "The outer while loop is the cooling schedule. As long as the "
         "temperature T is above Tmin, the inner for loop proposes inner "
         "neighbours. For each neighbour the change in cost, delta, is "
         "computed. If delta is negative the neighbour is better and it "
         "replaces the current tour. If delta is positive the neighbour "
         "is worse, but it is still accepted when a uniform random number "
         "is smaller than exp(-delta / T). That single if-statement is the "
         "Metropolis criterion. After the inner loop finishes, T is "
         "multiplied by alpha. Because alpha is 0.995, the temperature "
         "falls slowly: thousands of cooling steps occur before Tmin is "
         "reached, and tens of thousands of 2-opt moves are tried. The "
         "history list stores the best cost after every cooling step so "
         "that the cooling curve in the observation section can be drawn.")
    para(doc,
         "Three design choices are worth stating. First, the algorithm "
         "returns best, not current. Near the end of a run the current "
         "tour and the best tour are usually the same, but if a late "
         "uphill move is accepted we still keep the record. Second, the "
         "random seed is fixed inside the function so a reported tour can "
         "be reproduced. Third, the same code can be started from a random "
         "tour or from a greedy tour; only the start_tour argument changes.")

    heading(doc, "3.6 A greedy baseline, not the solver", 2)
    add_code(doc, extract(os.path.join(HERE, "tsp_sa.py"), 73, 85),
             "Snippet 6  -  nearest-neighbour construction")
    para(doc,
         "Nearest neighbour is the tour a person might sketch on a map: "
         "start in Jaipur and always drive to the closest city that has "
         "not been visited yet. It is fast and it already removes the "
         "worst crossings of a random permutation. It is not optimal. "
         "The last cities on the list are whatever was left over, and the "
         "closing edge back to Jaipur can be very long. The assignment "
         "asks for Simulated Annealing, so this routine is used only as a "
         "baseline. If SA cannot beat a greedy tour, the implementation "
         "or the cooling schedule is wrong.")

    heading(doc, "3.7 How the experiment is launched", 2)
    add_code(doc, extract(os.path.join(HERE, "tsp_sa.py"), 271, 290),
             "Snippet 7  -  one random start, one greedy start, extra seeds")
    para(doc,
         "run_experiment() is the script the report calls. It builds the "
         "matrix once, draws a random start (seed 1), builds a "
         "nearest-neighbour tour from Jaipur, then anneals the random "
         "start with seed 42. A second anneal starts from the greedy tour "
         "with a milder T0, because that tour is already short and does "
         "not need as much shaking. Three extra seeds are run so that a "
         "single lucky permutation is not mistaken for the method. All "
         "tours are rotated to begin at Jaipur before they are plotted or "
         "printed. The figures used in the next section are written into "
         "the figures folder by the same function.")

    para(doc,
         "To reproduce the numbers in this report from the terminal:",
         first_line=False)
    add_code(doc,
             "python tsp_sa.py\n"
             "python generate_report.py",
             "Snippet 8  -  commands")

    # ===== 4 OBSERVATION =====
    heading(doc, "4. Observation", 1)

    heading(doc, "4.1 The map of the instance", 2)
    para(doc,
         "Figure 1 is the raw instance: twenty-two points plotted by "
         "longitude and latitude. West is the Thar desert (Jaisalmer, "
         "Barmer). South is the Aravalli and tribal belt (Mount Abu, "
         "Udaipur, Dungarpur). East is the wildlife and Bharatpur side. "
         "North is Bikaner and the Shekhawati town of Mandawa. A good "
         "tour should look like a simple loop around this cloud of "
         "points. A bad tour looks like a scribble that crosses itself.")
    add_picture(doc, os.path.join(fig, "cities_map.png"), 6.1)
    caption(doc, "Figure 1. The twenty-two Rajasthan tourist locations used as TSP cities.")

    city_table = doc.add_table(rows=1 + len(CITIES), cols=4)
    city_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    fill_header_row(city_table.rows[0], ["No.", "Location", "Latitude (N)", "Longitude (E)"])
    for i, (name, lat, lon) in enumerate(CITIES):
        fill_row(city_table.rows[i + 1],
                 [i + 1, name, "%.4f" % lat, "%.4f" % lon],
                 shade="F4F6F8" if i % 2 == 0 else None)
    caption(doc, "Table 2. Names and coordinates of the tourist locations.")

    heading(doc, "4.2 A few pairwise distances", 2)
    para(doc,
         "Table 3 shows the haversine distance from Jaipur to every other "
         "stop. These numbers are the edge costs that leave the home city. "
         "They already hint at the geometry: Ajmer and Pushkar are cheap "
         "first hops, Jaisalmer and Barmer are expensive if they are "
         "visited as isolated out-and-back trips, and Bharatpur sits on "
         "the opposite side of the state. A sensible tour should pick up "
         "the eastern towns together and the desert towns together, rather "
         "than bouncing from Bharatpur to Jaisalmer and back.")

    d = data["dist"]
    dist_table = doc.add_table(rows=len(CITIES), cols=3)
    dist_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    fill_header_row(dist_table.rows[0], ["From", "To", "Distance (km)"])
    row_i = 1
    for j in range(1, len(CITIES)):
        fill_row(dist_table.rows[row_i],
                 ["Jaipur", CITIES[j][0], "%.1f" % d[0][j]],
                 shade="F4F6F8" if j % 2 == 0 else None)
        row_i += 1
    caption(doc, "Table 3. Haversine distances from Jaipur to the other twenty-one locations.")

    heading(doc, "4.3 Random start versus annealed tour", 2)
    para(doc,
         "The random initial tour has length %.2f km. After Simulated "
         "Annealing the same search, started from that tour, returns a "
         "cycle of %.2f km. That is a reduction of %.1f percent. Figure 2 "
         "puts the two cycles on the same scale. The left panel still "
         "contains long diagonals that cut across Rajasthan. The right "
         "panel is a single loop that follows the outline of the state: "
         "east through Alwar and Bharatpur, south through Ranthambore, "
         "Kota and Bundi, west along the southern forts and lakes, then "
         "through the desert and back down from Bikaner and Mandawa. "
         "The yellow marker is Jaipur, the start and the end."
         % (data["init_cost"], sa["cost"], improve))
    add_picture(doc, os.path.join(fig, "before_after.png"), 6.4)
    caption(doc, "Figure 2. Left: random initial cycle. Right: cycle returned by Simulated Annealing.")

    add_picture(doc, os.path.join(fig, "initial_tour.png"), 5.6)
    caption(doc, "Figure 3. Random initial tour (%.1f km)." % data["init_cost"])
    add_picture(doc, os.path.join(fig, "sa_tour.png"), 5.6)
    caption(doc, "Figure 4. Simulated Annealing tour (%.1f km), starting and ending at Jaipur." % sa["cost"])

    heading(doc, "4.4 Cooling curve", 2)
    para(doc,
         "Figure 5 plots two things against the cooling step. The blue "
         "curve is the best tour cost seen so far. The red curve is the "
         "temperature, drawn on a log scale. At the left the temperature "
         "is thousands of degrees and the best cost drops quickly: almost "
         "any 2-opt move that removes a long crossing is accepted, and "
         "many uphill moves are accepted as well. In the middle of the "
         "run the best-cost curve flattens into a staircase. That is the "
         "search sitting in a basin and only occasionally finding a "
         "shorter neighbour. On the far right the temperature is tiny, "
         "uphill moves are refused, and the curve is flat. The algorithm "
         "has frozen. The fact that the blue curve never rises is by "
         "construction: it records the best-so-far, not the current tour.")
    add_picture(doc, os.path.join(fig, "cooling_curve.png"), 6.3)
    caption(doc, "Figure 5. Best tour cost and temperature during geometric cooling.")

    heading(doc, "4.5 Comparison with the greedy tour", 2)
    para(doc,
         "Nearest neighbour, starting at Jaipur, builds a tour of %.2f km. "
         "That is already much better than a random permutation, which is "
         "expected: the greedy rule refuses the worst hops while unused "
         "cities remain. Simulated Annealing from a random start still "
         "beats this baseline by %.1f percent (%.2f km against %.2f km). "
         "When the annealer is itself started from the nearest-neighbour "
         "tour it reaches the same length, %.2f km. The two annealed "
         "cycles are the same loop in opposite directions. That is a "
         "useful observation: on this 22-city map SA does not need a "
         "clever start, and a greedy start leads to the same basin."
         % (data["nn_cost"], vs_nn, sa["cost"], data["nn_cost"], sa_nn["cost"]))
    add_picture(doc, os.path.join(fig, "nn_tour.png"), 5.6)
    caption(doc, "Figure 6. Nearest-neighbour tour (%.1f km), used only as a baseline." % data["nn_cost"])

    heading(doc, "4.6 Several random seeds", 2)
    para(doc,
         "Simulated Annealing is stochastic, so Table 4 repeats the run "
         "from independent random starts. Every annealed run returned the "
         "same cost, %.2f km, and the same cycle (or that cycle reversed). "
         "All of them beat the greedy construction. About one quarter of "
         "the proposed 2-opt moves were accepted. That is healthy: if "
         "every move were accepted the search would be a random walk; if "
         "almost none were accepted it would freeze too soon."
         % sa["cost"])

    rows_data = [
        ("Random start", data["init_cost"], "-", "-"),
        ("Nearest neighbour", data["nn_cost"], "-", "-"),
        ("SA  seed 42 (main)", sa["cost"], "%.3f" % sa["seconds"],
         "%.1f%%" % (100.0 * sa["accepted"] / sa["proposed"])),
        ("SA  from NN, seed 7", sa_nn["cost"], "%.3f" % sa_nn["seconds"],
         "%.1f%%" % (100.0 * sa_nn["accepted"] / sa_nn["proposed"])),
    ]
    for r in extra:
        rows_data.append((
            "SA  seed %s" % r["seed"],
            r["cost"],
            "%.3f" % r["seconds"],
            "%.1f%%" % (100.0 * r["accepted"] / r["proposed"]),
        ))

    res_table = doc.add_table(rows=1 + len(rows_data), cols=4)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    fill_header_row(res_table.rows[0],
                    ["Method", "Tour cost (km)", "Time (s)", "Moves accepted"])
    for i, row in enumerate(rows_data):
        fill_row(res_table.rows[i + 1],
                 [row[0],
                  ("%.2f" % row[1]) if isinstance(row[1], float) else row[1],
                  row[2], row[3]],
                 shade="F4F6F8" if i % 2 == 0 else None)
    caption(doc, "Table 4. Observed tour costs. Simulated Annealing is run from more than one seed.")

    heading(doc, "4.7 What the numbers are saying", 2)
    para(doc,
         "Three observations follow from the figures and from Table 4. "
         "First, the cost model is doing its job: once kilometres are "
         "treated as money, the annealer discovers the obvious geography "
         "of Rajasthan and stops crossing the state. Second, Simulated "
         "Annealing is stronger than both a random cycle and a one-pass "
         "greedy cycle on this instance. Third, the method is stable. "
         "Changing the seed did not change the cost at all: every run "
         "froze on the same cycle. For a week-long family visit that is "
         "enough. The relatives will not drive an extra thousand "
         "kilometres because the random number generator was unlucky.")
    para(doc,
         "A limitation should be stated as well. The search does not "
         "prove optimality. A still-shorter cycle may exist. Road "
         "distance, one-way restrictions and the time spent inside each "
         "city are also missing. If the family later drops Mount Abu or "
         "adds Ranakpur as a half-day halt, the same program can be "
         "re-run; only the city list has to change.")

    # ===== 5 RESULT =====
    heading(doc, "5. Result", 1)
    para(doc,
         "The assignment is solved. Twenty-two important tourist locations "
         "of Rajasthan were modelled as a complete symmetric TSP. Travel "
         "cost was taken as the haversine distance in kilometres. "
         "Simulated Annealing with a 2-opt neighbourhood, geometric "
         "cooling (T0 = %.0f, alpha = %s, Tmin = %s) and the Metropolis "
         "rule produced a closed tour of %.2f km starting and ending at "
         "Jaipur. A random initial tour of the same cities cost %.2f km. "
         "A nearest-neighbour tour cost %.2f km. The annealed tour is "
         "therefore the plan that should be given to the visiting relatives."
         % (sa["t0"], sa["alpha"], sa["t_min"], sa["cost"],
            data["init_cost"], data["nn_cost"]))

    para(doc, "Recommended tour (main SA run, seed 42)",
         first_line=False, bold=True, space_after=4)
    para(doc, format_tour(data["sa_tour"], CITIES),
         first_line=False, size=11, space_after=10)

    # numbered stop table
    tour = data["sa_tour"]
    stop_table = doc.add_table(rows=1 + len(tour) + 1, cols=4)
    stop_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    fill_header_row(stop_table.rows[0],
                    ["Stop", "Location", "Leg (km)", "Distance so far (km)"])
    running = 0.0
    for i, city_idx in enumerate(tour):
        nxt = tour[(i + 1) % len(tour)] if i < len(tour) - 1 else tour[0]
        if i == 0:
            fill_row(stop_table.rows[1],
                     [1, CITIES[city_idx][0], "-", "0.00"],
                     shade="FFF3CD")
        else:
            prev = tour[i - 1]
            leg = d[prev][city_idx]
            running += leg
            fill_row(stop_table.rows[i + 1],
                     [i + 1, CITIES[city_idx][0], "%.1f" % leg, "%.1f" % running],
                     shade="F4F6F8" if i % 2 == 0 else None)
    last_leg = d[tour[-1]][tour[0]]
    running += last_leg
    fill_row(stop_table.rows[len(tour) + 1],
             ["return", "Jaipur", "%.1f" % last_leg, "%.1f" % running],
             shade="FFF3CD")
    caption(doc, "Table 5. Ordered stops of the recommended tour and the running distance.")

    para(doc,
         "The result can be read in one sentence. Do not send the family "
         "on a random sightseeing order, and do not trust the first greedy "
         "loop drawn on a map. Run Simulated Annealing on the distance "
         "matrix, keep the best cycle, and start that cycle in Jaipur. "
         "On the instance used here the family saves roughly %.0f km "
         "against a random plan and roughly %.0f km against nearest "
         "neighbour. That is the cost-effective tour asked for in the "
         "question."
         % (data["init_cost"] - sa["cost"], data["nn_cost"] - sa["cost"]))

    heading(doc, "5.1 Conclusion", 2)
    para(doc,
         "TSP on twenty-two Rajasthan cities is already too large for "
         "brute force. Simulated Annealing is a suitable local-search "
         "answer: the state is a permutation, the move is 2-opt, the "
         "acceptance rule is Metropolis, and the schedule is geometric "
         "cooling. With travel cost taken as distance, the method returns "
         "a short, geographically clean cycle that can be used as a "
         "one-week tour plan. The implementation is reproducible from "
         "tsp_sa.py, and the figures in this report were generated by "
         "the same run that produced the numbers.")

    heading(doc, "5.2 Files submitted", 2)
    files = [
        ("cities.py", "List of 22 locations with latitude and longitude."),
        ("tsp_sa.py", "Distance model, Simulated Annealing, plots, main program."),
        ("generate_report.py", "Builds this Word document from a live run."),
        ("requirements.txt", "matplotlib, python-docx."),
        ("figures/", "cities_map, initial tour, SA tour, NN tour, cooling curve."),
        ("Lab_Assignment_2_TSP_Simulated_Annealing.docx", "This report."),
    ]
    ftab = doc.add_table(rows=1 + len(files), cols=2)
    ftab.alignment = WD_TABLE_ALIGNMENT.CENTER
    fill_header_row(ftab.rows[0], ["File", "Role"])
    for i, (name, role) in enumerate(files):
        fill_row(ftab.rows[i + 1], [name, role], center=False,
                 shade="F4F6F8" if i % 2 == 0 else None)

    doc.save(OUT)
    print("Wrote", OUT)
    print("Main SA cost: %.2f km" % sa["cost"])


if __name__ == "__main__":
    build()
