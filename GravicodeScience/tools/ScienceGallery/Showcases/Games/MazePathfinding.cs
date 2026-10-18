using Gravicode.Science.GraviGraph;
using Gravicode.Science.GraviGraph.Algorithms;
using Gravicode.Science.GraviNum;
using Matplotlib.Net.Rendering;
using ScienceGallery.Models;
using ScienceGallery.Services;
using Visual.Net.Core.Colors;

namespace ScienceGallery.Showcases.Games;

/// <summary>Generating a maze, solving it, and computing the distance field game monsters use to chase the player.</summary>
public sealed class MazePathfinding : Showcase
{
    public override string Title => "Build a maze, solve it, and let every monster find the player";
    public override Domain Domain => Domain.Games;
    public override GraviLibrary[] Libraries => [GraviLibrary.GraviGraph, GraviLibrary.GraviNum];
    public override string Summary => "A depth-first maze as a graph, its shortest path, and one breadth-first search as a flow field.";
    public override string Scenario =>
        "A dungeon game carves a fresh maze for each level with a randomised depth-first search, which makes long " +
        "winding corridors with exactly one route between any two rooms. As a graph, the maze is solved by a " +
        "shortest-path search. For enemies, one breadth-first search from the player gives every cell its step " +
        "distance; each monster then just moves to a neighbour with a smaller number, so a hundred monsters cost " +
        "one search instead of a hundred.";

    public override void Run(ShowcaseOutput output)
    {
        const int width = 30, height = 20;                  // cells
        int Cell(int x, int y) => y * width + x;

        // A perfect maze by randomised depth-first search: walk to a random unvisited neighbour, knocking
        // the wall down; back up when stuck. Every knocked-down wall becomes a graph edge.
        var rng = new GraviRandom(5);
        var maze = new Graph(width * height, directed: false);
        var visited = new bool[width * height];
        var stack = new Stack<(int X, int Y)>();
        stack.Push((0, 0));
        visited[0] = true;
        (int dx, int dy)[] steps = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        while (stack.Count > 0)
        {
            var (x, y) = stack.Peek();
            var options = steps.Select(s => (X: x + s.dx, Y: y + s.dy))
                .Where(n => n.X >= 0 && n.X < width && n.Y >= 0 && n.Y < height && !visited[Cell(n.X, n.Y)]).ToArray();
            if (options.Length == 0) { stack.Pop(); continue; }
            var next = options[rng.Next(options.Length)];
            maze.AddEdge(Cell(x, y), Cell(next.X, next.Y));
            visited[Cell(next.X, next.Y)] = true;
            stack.Push(next);
        }

        var entrance = Cell(0, 0);
        var exit = Cell(width - 1, height - 1);
        var path = GraphAlgorithms.ShortestPath(maze, entrance, exit);

        // The flow field: steps from the player (here, the middle of the maze) to every cell.
        var player = Cell(width / 2, height / 2);
        var stepsToPlayer = GraphAlgorithms.HopDistances(maze, player);
        var field = new double[height, width];
        for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++) field[y, x] = stepsToPlayer[Cell(x, y)];
        var deadEnds = Enumerable.Range(0, width * height).Count(c => maze.Degree(c) == 1);

        var figure = GalleryCharts.Figure(10.5, 6.6);
        var ax = figure.AddSubplot();
        ax.ImShow(field, colormap: "viridis", extent: (0, width, height, 0), alpha: 0.85);
        ax.AddColorbar("steps to the player");
        // Walls: draw a segment wherever two neighbouring cells are not connected, plus the outer border.
        var wall = Color.Parse("#FFFFFF");
        for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
            {
                if (x + 1 < width && !maze.HasEdge(Cell(x, y), Cell(x + 1, y))) ax.Plot([x + 1, x + 1], [y, y + 1], color: wall, lineWidth: 2.2);
                if (y + 1 < height && !maze.HasEdge(Cell(x, y), Cell(x, y + 1))) ax.Plot([x, x + 1], [y + 1, y + 1], color: wall, lineWidth: 2.2);
            }
        ax.Plot([0, width, width, 0, 0], [0, 0, height, height, 0], color: wall, lineWidth: 3);
        ax.Plot(path.Select(c => c % width + 0.5).ToArray(), path.Select(c => c / width + 0.5).ToArray(), "shortest route, entrance to exit",
                Color.Parse("#D9363E"), lineWidth: 2.4);
        ax.Scatter([width / 2 + 0.5], [height / 2 + 0.5], "player", Color.Parse("#E8A317"), sizes: [180], marker: MarkerStyle.Star);
        ax.SetAspect(1.0).AxisOff().SetTitle("Maze coloured by steps to the player (star); red: shortest route, top-left to bottom-right");
        output.Chart(figure);

        output.Metric("Cells / corridors", $"{width * height} / {maze.EdgeCount}", "a perfect maze: corridors = cells − 1");
        output.Metric("Shortest route", $"{path.Count - 1} steps", $"straight-line minimum would be {width + height - 2}");
        output.Metric("Farthest cell from the player", $"{stepsToPlayer.Max()} steps");
        output.Metric("Dead ends", deadEnds.ToString(), "cells with one way in");
        output.Line($"The route winds {(path.Count - 1.0) / (width + height - 2):F1}× longer than the open-floor distance. One search from the player " +
                    "labelled all 600 cells, enough to steer every monster on the level.");
    }
}
