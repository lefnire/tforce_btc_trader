import React, { Component } from 'react';
import _ from 'lodash';
import ReactTable from 'react-table';
require('react-table/react-table.css');
const d3 = require('d3');
const uuidv4 = require('uuid/v4');
const d3Tip = require('d3-tip');

class App extends Component {
  constructor() {
    super();
    this.state = {
      data: null,
      showSignals: false
    };
  }

  componentDidMount() {
    // fetch('http://localhost:5000').then(res => res.json()).then(data => {
    fetch('dumps/alex.json').then(res => res.json()).then(data => {
      data.forEach(d => {
        d.hypers = _.transform(d.hypers, (m,v,k) => {
          m[k.replace(/\./g, '_')] = typeof v == 'boolean' ? ~~v : v;
        });
        d.unique_sigs = _.uniq(d.actions).length;
        d.id = uuidv4();
      });
      this.orig_data = data;
      this.setState({data}, () => {
        this.mountChart(data);
      });
    });
  }

  defaultFilterMethod = (filter, row, column) => {
    const id = filter.pivotId || filter.id;
    const col = row[id],
      txt = filter.value;
    if (col === undefined) return true;
    if (~(""+col).indexOf(txt)) return true;
    if (~txt.indexOf('>')) {
      return +col > +txt.substring(1);
    } else if (~txt.indexOf('<')) {
      return +col < +txt.substring(1);
    }
    return false;
  };

  onFilteredChange = () => {
    this.mountChart(_.map(this.refs.reactTable.state.sortedData, '_original'));
  };

  renderTable = () => {
    let {data} = this.state;
    if (!data) return;
    let columns = ['unique_sigs', 'reward_avg'/*, 'source'*/].map(k => ({Header: k, accessor: k}));
    columns = columns.concat(
      _(data)
        .map(d => _.keys(d.hypers))
        .flatten()
        .uniq()
        .without('net_type', 'step_optimizer_type')
        .map(k => ({Header: k, accessor: 'hypers.' + k}))
        .value()
    );

    return <ReactTable
      ref='reactTable'
      minRows={2}
      data={data}
      columns={columns}
      filterable={true}
      defaultFilterMethod={this.defaultFilterMethod}
      onFilteredChange={_.debounce(this.onFilteredChange, 400)}
      defaultSorted={[{id:'reward_avg', desc:true}]}
      defaultPageSize={10}
    />
  };

  render() {
    const width = window.innerWidth - 50,
      height = window.innerHeight - 50;
    const {showSignals} = this.state;
    return (
      <div>
        {this.renderTable()}
        <svg id='rewards' width={width} height={height / 2} />
        <svg id='signals' width={width} height={showSignals? (height / 2) : 0} />
      </div>
    );
  }

  mountChart = (data) => {
    let svg = d3.select("svg#rewards"),
      margin = {top: 20, right: 20, bottom: 30, left: 50},
      width = +svg.attr("width") - margin.left - margin.right,
      height = +svg.attr("height") - margin.top - margin.bottom;

    svg.select('g').remove(); // start clean
    let g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let rewards = data.map(d => {
      return d.rewards.map((v,i) => ({y:_.clamp(v,-300,300), x:i, parent:d})); // note clamp so we don't break the graph
    });
    let all_rewards = _.flatten(rewards);

    let colorScale = d3.scaleOrdinal(d3.schemeCategory10)
      .domain([0, data.length]);

    let x = d3.scaleLinear()
      .rangeRound([0, width])
      .domain([0, d3.max(rewards, r => r.length)]);
    let y = d3.scaleLinear()
      .rangeRound([height, 0])
      .domain(d3.extent(all_rewards, d => d.y));

    let axes = {
      x: d3.axisBottom(x),
      y: d3.axisLeft(y)
    };
    let axes_g = {
      x: g.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(axes.x),
      y: g.append("g")
        .call(axes.y)
    };
    axes_g.y.append("text")
      .attr("fill", "#000")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", "0.71em")
      .attr("text-anchor", "end")
      .text("Reward");

    let line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y));

    let radius = 2.5,
      lineWidth = 1.5;

    rewards.forEach((row, i) => {
      g.append("path")
        .datum(row)
        .classed('line', true)
        .attr("fill", "none")
        .attr("stroke", () => colorScale(i))
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("d", line)
        .on("click", d => {
          this.clickedDatum = d[0].parent;
          this.setState({showSignals: true}, this.mountSignals);
        })
    });

    // Add a 0-line
    let zeroLine = [
      {x:0,y:0},
      {x:d3.max(rewards, r => r.length), y:0}
    ];
    g.datum(zeroLine)
      .append('path')
      .classed('zero-line', true)
      .attr("fill", "none")
      .attr("stroke", 'black')
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("d", line)

    let tip = d3Tip()
      .attr('class', 'd3-tip')
      .direction('sw')
      .html(d => {
        let str = `
          reward: ${d.y}<br/>
          reward_avg: ${d.parent.reward_avg}<br/>
          source: ${d.parent.source}<br/>
          uniques: ${_.uniq(d.parent.actions).length}<br/><br/>
        `;

        return str + JSON.stringify(d.parent.hypers, null, 2).replace(/\n/g,'<br/>')
      });

    g.call(tip);
    g.selectAll("dot").data(all_rewards)
      .enter()
        .append("circle")
        .classed('dot', true)
        .attr("r", radius)
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .on("mouseover", tip.show)
        .on("mouseout", tip.hide)

    let zoom = d3.zoom()
      .scaleExtent([0,500])
      .on("zoom", zooming);
    svg.call(zoom);
    function zooming() {
      g.selectAll('.dot')
        .attr("transform", d3.event.transform)
        .attr('r', radius/d3.event.transform.k);
      g.selectAll('.line,.zero-line')
        .attr("transform", d3.event.transform)
        .attr("stroke-width", lineWidth/d3.event.transform.k);
      axes_g.x.call(axes.x.scale(d3.event.transform.rescaleX(x)));
      axes_g.y.call(axes.y.scale(d3.event.transform.rescaleY(y)));
    }
  };

  mountSignals = () => {
    let {actions, prices} = this.clickedDatum;
    let svg = d3.select("svg#signals");
    svg.select('g').remove(); // start fresh
    let margin = {top: 20, right: 20, bottom: 30, left: 50},
      width = +svg.attr("width") - margin.left - margin.right,
      height = +svg.attr("height") - margin.top - margin.bottom,
      g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let x = d3.scaleLinear()
      .rangeRound([0, width])
      .domain([0, prices.length]);
    let y = d3.scaleLinear()
      .rangeRound([height, 0])
      .domain(d3.extent(prices, d => d));

    let axes = {
      x: d3.axisBottom(x),
      y: d3.axisLeft(y)
    };
    let axes_g = {
      x: g.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(axes.x),
      y: g.append("g")
        .call(axes.y)
    };
    axes_g.y.append("text")
      .attr("fill", "#000")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", "0.71em")
      .attr("text-anchor", "end")
      .text("Price");

    let line = d3.line()
        .x((d,i) => x(i))
        .y(d => y(d));

    g.append("path")
      .datum(prices)
      .classed('line', true)
      .attr("fill", "none")
      .attr("stroke", 'steelBlue')
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("stroke-width", .5)
      .attr("d", line);

    g.selectAll("dot").data(prices)
      .enter()
        .append("circle")
        .classed('dot', true)
        .style('fill', (d,i) => actions[i] < 0 ? 'red' : actions[i] > 0 ? 'green' : 'rgba(0,0,0,0)')
        .attr("r", 2)
        .attr("cx", (d,i) => x(i))
        .attr("cy", d => y(d));
    let zoom = d3.zoom()
      .scaleExtent([0,500])
      .on("zoom", zooming);
    svg.call(zoom);
    function zooming() {
      g.selectAll('.line,.zero-line,.dot').attr("transform", d3.event.transform);
      axes_g.x.call(axes.x.scale(d3.event.transform.rescaleX(x)));
      axes_g.y.call(axes.y.scale(d3.event.transform.rescaleY(y)));
    }
  };
}

export default App;
