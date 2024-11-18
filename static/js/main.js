import { formatNodes, formatEdges } from './vis_tree_data.js'

// Setup.
const socket = io()
const container = document.getElementById('mcts-tree')
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
const network = new vis.Network(container, data, options)

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
