import { formatNodes, formatEdges, renderBoard } from './vis_tree_data.js'

// Set up socket.
const socket = io()

// Set up tree as network.
const tree = document.getElementById('mcts-tree')
const data = {
    nodes: new vis.DataSet(),
    edges: new vis.DataSet(),
}
const options = {
    layout: {
        hierarchical: {
            direction: 'UD', // Up-Down layout
            sortMethod: 'directed',
        },
    },
    interaction: {
        hover: true,
    },
    physics: false,
}
const network = new vis.Network(tree, data, options)

// Set up board.
const boardElement = document.getElementById('board')

// Render board data for clicked node.
network.on('click', function (params) {
    if (params.nodes.length > 0) {
        const nodeId = params.nodes[0]
        const nodeData = data.nodes.get(nodeId)
        console.log(`Node ID: ${nodeId}`)
        console.log(nodeData)
        renderBoard(boardElement, nodeData.board)
    }
})

// Define socket connection and update behavior.
socket.on('connect', () => {
    console.log('Connected to server')
    socket.emit('update_request', {}) // Request update from server
})

socket.on('update_response', (treeData) => {
    console.log('Update from server:', treeData)
    data.nodes.clear()
    data.edges.clear()
    const transformedNodes = formatNodes(treeData)
    const transformedEdges = formatEdges(treeData)
    data.nodes.add(transformedNodes)
    data.edges.add(transformedEdges)
})
