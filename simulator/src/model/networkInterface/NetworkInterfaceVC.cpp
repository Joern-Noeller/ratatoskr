////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2018 Jan Moritz Joseph
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////

#include "NetworkInterfaceVC.h"

NetworkInterfaceVC::NetworkInterfaceVC(sc_module_name nm, Node* node) :
		NetworkInterface(nm, node) {
	this->id = node->id;
	this->dbid = rep.registerElement("ProcessingElement", id);
	this->node = node;
	this->flowControlOut = new std::vector<bool>(1, true);
	this->tagOut = new std::vector<int>(1, 0);
	this->emptyOut = new std::vector<bool>(1, true);

	this->flitPortContainer = new FlitPortContainer(
			("NI_FLIT_CONTAINER" + std::to_string(id)).c_str());
	this->packetPortContainer = new PacketPortContainer(
			("NI_PACKET_CONTAINER" + std::to_string(id)).c_str());

}

void NetworkInterfaceVC::receivePacket() {
	LOG(global.verbose_pe_function_calls,
			"NI" << id <<"(Router"<<node->id<< ")t- receive()");



	if (packetPortContainer->portValidIn.posedge()) {

		Packet* p = packetPortContainer->portDataIn.read();
		p->toTransmit.resize(global.flitsPerPacket);

		Flit* headFlit;
		for (int i = 0; i < global.flitsPerPacket; i++) {
			FlitType type;
			if (i % global.flitsPerPacket == 0) {
				type = FlitType::HEAD;
			} else if (i % global.flitsPerPacket == global.flitsPerPacket - 1) {
				type = FlitType::TAIL;
			} else {
				type = FlitType::BODY;
			}
			Flit* current_flit = new Flit(type, i % global.flitsPerPacket, p,
					p->trafficTypeId, sc_time_stamp().to_double());
			if (type == FlitType::HEAD) {
				headFlit = current_flit;
			}
			current_flit->headFlit = headFlit;
			p->toTransmit.at(global.flitsPerPacket - i - 1) = current_flit;
		}
		packet_send_queue.push(p);
	}
}

NetworkInterfaceVC::~NetworkInterfaceVC() {

}

void NetworkInterfaceVC::initialize() {

	flitPortContainer->portValidOut.write(false);
	flitPortContainer->portFlowControlOut.write(flowControlOut);
	flitPortContainer->portTagOut.write(tagOut);
	flitPortContainer->portEmptyOut.write(emptyOut);

	SC_METHOD(thread);
	sensitive << clk.pos() << clk.neg();

	SC_METHOD(receiveFlit);
	sensitive << flitPortContainer->portValidIn.pos();

	SC_METHOD(receivePacket);
	sensitive << packetPortContainer->portValidIn.pos();
}

void NetworkInterfaceVC::bind(Connection* con, SignalContainer* sigContIn,
		SignalContainer* sigContOut) {
	if (con == 0) {
		packetPortContainer->bind(sigContIn, sigContOut);
	} else {
		flitPortContainer->bind(sigContIn, sigContOut);
	}
}

void NetworkInterfaceVC::thread() {
	LOG(global.verbose_pe_function_calls,
			"NI" << node->idType <<"(Node"<<node->id<< ")\t- send_data_process()");

	if (clk.posedge()) {
		if (!packet_send_queue.empty()) {
			if (flitPortContainer->portFlowControlIn.read()->at(0)) {
				//generate flit and send it.
				//cout << "generating flit" << endl;
				Packet* p = packet_send_queue.front();

				Flit* current_flit = p->toTransmit.back();
				current_flit->injectionTime = sc_time_stamp().to_double();
				p->toTransmit.pop_back();
				p->inTransmit.push_back(current_flit);

				rep.reportEvent(dbid, "pe_send_flit",
						std::to_string(current_flit->id));

				if (p->toTransmit.empty()) {
					packet_send_queue.pop();
				}

				flitPortContainer->portValidOut.write(true);
				flitPortContainer->portDataOut.write(current_flit);
				flitPortContainer->portVcOut.write(0);

				LOG(
						(global.verbose_pe_send_head_flit
								&& current_flit->type == HEAD)
								|| global.verbose_pe_send_flit,
						"NI" << node->idType <<"(Node"<<node->id<< ")\t- Send Flit " << *current_flit);
			} else {
				LOG(global.verbose_pe_throttle,
						"NI" << node->idType <<"(Node"<<node->id<< ")\t- Waiting for Router!");
			}

		}

		if (!packet_recv_queue.empty()) {

			if (packetPortContainer->portFlowControlIn.read()) {
				Packet* p = packet_recv_queue.front();
				packet_recv_queue.pop();

				packetPortContainer->portValidOut.write(true);
				packetPortContainer->portDataOut.write(p);

			}

		}

	} else if (clk.negedge()) {
		flitPortContainer->portValidOut.write(false);
		packetPortContainer->portValidOut.write(false);
	}
}

void NetworkInterfaceVC::receiveFlit() {
	LOG(global.verbose_pe_function_calls,
			"NI" << node->idType <<"(Node"<<node->id<< ")\t- receive_data_process()");

	//to debug: simply print the received flit
	if (flitPortContainer->portValidIn.posedge()) {

		//if no flit was send: ABP or with NULL-check?
		Flit* received_flit = flitPortContainer->portDataIn.read();
		Packet* p = received_flit->packet;

		// generate packet statistics. in case of synthetic traffic only for run phase
		if ((float)global.synthetic_start_measurement_time
				<= (sc_time_stamp().to_double() / (float) 1000)) {
			report.latencyFlit.sample(
					sc_time_stamp().to_double() - received_flit->injectionTime);
			if (received_flit->type == TAIL) {
				report.latencyPacket.sample(
						sc_time_stamp().to_double()
								- received_flit->headFlit->creationTime);
				report.latencyNetwork.sample(
						sc_time_stamp().to_double()
								- received_flit->headFlit->injectionTime);
			}
		}

		std::vector<Flit*>::iterator position = std::find(p->inTransmit.begin(),
				p->inTransmit.end(), received_flit);
		if (position != p->inTransmit.end())
			p->inTransmit.erase(position);

		p->transmitted.push_back(received_flit);
		rep.reportEvent(dbid, "pe_receive_flit",
				std::to_string(received_flit->id));

		LOG(
				(global.verbose_pe_receive_tail_flit
						&& received_flit->type == TAIL)
						|| global.verbose_pe_receive_flit,
				"NI" << node->idType <<"(Node"<<node->id<< ")\t- Receive Flit " << *received_flit);

		LOG(
				received_flit->type == TAIL
						&& (!p->toTransmit.empty() || !p->inTransmit.empty()),
				"NI" << node->idType <<"(Node"<<node->id<< ")\t- Received Tail Flit, but still missing flits! " << *received_flit);

		//if a complete packet is recieved, notify tasks
		if (p->toTransmit.empty() && p->inTransmit.empty()) {
//				NetracePacket* np = dynamic_cast<NetracePacket*>(p);
//
//				if (np && np->dst != id) {
//					cout << "RECIEVED WRONG FLIT" << endl;
//				}
//				if (np && global.verbose_netrace_router_receive) {
//					cout << "PE " << id << " recieved a packet @ " << sc_time_stamp() << ":";
//					Netrace netrace;
//					netrace.nt_print_packet(np->netracePacket);
//				}

			//TODO
			//trafficPool->receive(p);

			packet_recv_queue.push(p);

		}
	}
	// build packet from flit, send packet to application (via TLM?)
}
