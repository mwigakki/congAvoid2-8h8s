/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4 = 0x800;
#define LENGTH 50000 //寄存器位数，保证存够100s数据

/*************************************************************************
*********************** H E A D E R S  ***********************************
*************************************************************************/

typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;

header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

struct metadata {
    /* empty */
}

struct headers {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
}

/*************************************************************************
*********************** P A R S E R  ***********************************
*************************************************************************/

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition accept;
    }

}

/*************************************************************************
************   C H E C K S U M    V E R I F I C A T I O N   *************
*************************************************************************/

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}


/*************************************************************************
**************  I N G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
    // 先试试 10000个空间
    register<bit<32>>(2) reg_idx_and_cnt; 
    register<bit<48>>(LENGTH) reg_ingress_global_timestamp; 
    register<bit<48>>(LENGTH) reg_egress_global_timestamp;  

    counter(8, CounterType.packets) pkt_counter;    // define a packets counter with 8 size;
    // register <bit<32>>(8) pkt_counter;			// register<T>
    // unfortunately, read register by grpc is not supported for now.

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {
        standard_metadata.egress_spec = port;
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    apply {
        if (hdr.ipv4.isValid()) {
            bit<8> tmp = hdr.ipv4.dstAddr[15:8];
            bit<32> idx = (bit<32>)(tmp - 1);
            pkt_counter.count(idx);

            ipv4_lpm.apply();

            bit<32> idx1;//字节数
            bit<32> cnt;//新的字节数
            reg_idx_and_cnt.read(idx1, 0);
            reg_idx_and_cnt.read(cnt, 1);
            if (cnt == 0){
                reg_ingress_global_timestamp.write(idx1, standard_metadata.ingress_global_timestamp);
                reg_egress_global_timestamp.write(idx1, standard_metadata.egress_global_timestamp);
                idx1 = idx1 + 1;
                cnt = cnt + 1;
                reg_idx_and_cnt.write(0, idx1);
                reg_idx_and_cnt.write(1, cnt);
            } else if (cnt == 10){
                reg_idx_and_cnt.write(1, 0); // 在第 1 个位置 cnt 上写上0
            } else {
                cnt = cnt + 1;
                reg_idx_and_cnt.write(1, cnt);
            }
        }
    }
}

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    register<bit<32>>(2) reg_idx_and_cnt; 
    
// register_read reg_deq_qdepth
    register<bit<32>>(LENGTH) reg_enq_timestamp;    
    register<bit<19>>(LENGTH) reg_enq_qdepth;     
    register<bit<32>>(LENGTH) reg_deq_timedelta;     
    register<bit<19>>(LENGTH) reg_deq_qdepth;   

    apply { 
        bit<32> idx;//字节数
        bit<32> cnt;//新的字节数
        reg_idx_and_cnt.read(idx, 0);
        reg_idx_and_cnt.read(cnt, 1);
        if (cnt == 0){
            reg_enq_timestamp.write(idx, standard_metadata.enq_timestamp);  // (bit<32>)
            reg_enq_qdepth.write(idx, standard_metadata.enq_qdepth);
            reg_deq_timedelta.write(idx, standard_metadata.deq_timedelta);
            reg_deq_qdepth.write(idx, standard_metadata.deq_qdepth);
            idx = idx + 1;
            cnt = cnt + 1;
            reg_idx_and_cnt.write(0, idx);
            reg_idx_and_cnt.write(1, cnt);
        } else if (cnt == 10){
            reg_idx_and_cnt.write(1, 0); // 在第 1 个位置 cnt 上写上0
        } else {
            cnt = cnt + 1;
            reg_idx_and_cnt.write(1, cnt);
        }
    }
}
/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers  hdr, inout metadata meta) {
     apply {
        update_checksum(
        hdr.ipv4.isValid(),
            { hdr.ipv4.version,
              hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;
