/*******************************************************************************
 * Copyright (C) 2018 joseph
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/
#include <vector>
#include "NoC.h"

NoC::NoC(sc_module_name nm)
{
    dbid = rep.registerElement("NoC", 0);
    networkParticipants.resize(globalResources.nodes.size());
    signalContainers.resize(globalResources.connections.size()*2);
    links.resize(globalResources.connections.size()*2);

    createTrafficPool();

    //create a clock for every node type
    std::vector<std::unique_ptr<sc_clock>> clocks(globalResources.nodeTypes.size());
    for (const auto& nodeType : globalResources.nodeTypes) {
        clocks.at(nodeType->id) = std::make_unique<sc_clock>(("NodeType"+std::to_string(nodeType->id)+"Clock").c_str(),
                nodeType->clockDelay, SC_NS);
    }

    createNetworkParticipants(clocks);
    createSigContainers();
    createLinks(clocks);
    runNoC();
}

void NoC::createTrafficPool()
{
    if (globalResources.benchmark=="task") {
        tp = std::make_unique<TaskPool>();
    }
    else if (globalResources.benchmark=="synthetic") {
        tp = std::make_unique<SyntheticPool>();
    }
    else {
        FATAL("Please specify correct benchmark type");
    }
}

void NoC::createNetworkParticipants(const std::vector<std::unique_ptr<sc_clock>>& clocks)
{
    unsigned long pe_size = 0;
    for (Node& n : globalResources.nodes) {
        if (n.type->model=="RouterVC") {
            std::string name = "router_"+std::to_string(n.id);
            RouterVC* r = new RouterVC(name.c_str(), n);
            r->clk(*clocks.at(n.type->id));
            networkParticipants.at(n.id) = dynamic_cast<NetworkParticipant*>(r);
            delete r;
        }
        else if (n.type->model=="ProcessingElement") {
            // Creating an network interface.
            std::string ni_name = "ni_"+std::to_string(n.id);
            Node* n_ptr = &n;
            NetworkInterfaceVC* ni = new NetworkInterfaceVC(ni_name.c_str(), n_ptr);
            ni->clk(*clocks.at(n.type->id));
            networkParticipants.at(n.id) = dynamic_cast<NetworkParticipant*>(ni);

            // Creating a processing element.
            std::string pe_name = "pe_"+std::to_string(n.id);
            ProcessingElementVC* pe = new ProcessingElementVC(pe_name.c_str(), n, tp.get());
            std::unique_ptr<PacketSignalContainer> sig1 = std::make_unique<PacketSignalContainer>(
                    ("packetSigCon1_"+std::__cxx11::to_string(n.id)).c_str());
            std::unique_ptr<PacketSignalContainer> sig2 = std::make_unique<PacketSignalContainer>(
                    ("packetSigCon2_"+std::__cxx11::to_string(n.id)).c_str());
            ni->bind(nullptr, sig1.get(), sig2.get());
            pe->bind(nullptr, sig2.get(), sig1.get());
            networkParticipants.push_back(dynamic_cast<NetworkParticipant*>(pe));
            signalContainers.push_back(move(sig1));
            signalContainers.push_back(move(sig2));
            tp->processingElements.at(n.id%tp->processingElements.size()) = move(
                    std::make_unique<ProcessingElementVC>(pe));

            pe_size++;

            delete ni;
            delete pe;
        }
    }
    tp->processingElements.resize(pe_size);
}

void NoC::createSigContainers()
{
    for (int i = 0; i<signalContainers.size(); i++) {
        signalContainers.at(i) = std::make_unique<FlitSignalContainer>(
                ("flitSigCont_"+std::__cxx11::to_string(i)).c_str());
    }
}

void NoC::createLinks(const std::vector<std::unique_ptr<sc_clock>>& clocks)
{
    for (int i = 0; i<globalResources.connections.size(); i += 2) {
        Connection& c = globalResources.connections.at(i);

        if (c.nodes.size()==2) { //might extend to bus architecture
            links.at(i) = std::make_unique<Link>(
                    ("link_"+std::to_string(i)+"_Conn_"+std::to_string(c.id)).c_str(), c, i);
            links.at(i+1) = std::make_unique<Link>(
                    ("link_"+std::to_string(i+1)+"_Conn_"+std::to_string(c.id)).c_str(), c,
                    i+1);

            Node& node1 = globalResources.nodes.at(c.nodes.at(0));
            Node& node2 = globalResources.nodes.at(c.nodes.at(1));

            std::unique_ptr<Connection> c_ptr = std::make_unique<Connection>(c);
            networkParticipants.at(node1.id)->bind(c_ptr.get(), signalContainers.at(i).get(),
                    signalContainers.at(i+1).get());
            networkParticipants.at(node2.id)->bind(c_ptr.get(), signalContainers.at(i+1).get(),
                    signalContainers.at(i).get());

            links.at(i)->classicPortContainer->bindOpen(signalContainers.at(i).get());
            links.at(i+1)->classicPortContainer->bindOpen(signalContainers.at(i+1).get());
            links.at(i)->clk(*clocks.at(node1.type->id));
            links.at(i+1)->clk(*clocks.at(node2.type->id));
        }
        else {
            FATAL("Unsupported number of endpoints in connection " << c.id);
        }
    }
}

void NoC::runNoC()
{
    for (auto& r : networkParticipants) {
        r->initialize();
    }
    tp->start();
}

NoC::~NoC()
{
    for (auto& r : networkParticipants) {
        delete r;
    }
}
